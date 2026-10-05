"""Stdio tools for a host LLM: local/shared corpus, 2026+ discovery, DOI hydration."""
from .backends import Remote
from .live import OpenAlex
from .retriever import Retriever,normalize_doi


def serve_mcp(settings):
    try:
        from mcp.server import MCPServer
    except ImportError:
        try: from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError: raise RuntimeError('Install the optional MCP SDK: python -m pip install "mcp>=1.26,<3"') from None
    server=MCPServer('Urban Planning Literature')
    remote=Remote(settings['service_url'],settings.get('service_token_env','PLANNING_SERVICE_TOKEN')) if settings.get('backend')=='remote' else None
    live=OpenAlex(settings.get('openalex_cache'),settings.get('openalex_key_env','OPENALEX_API_KEY'))

    @server.tool()
    def search_planning_corpus(query:str,from_year:int=2026,to_year:int|None=None,limit:int=8,fts_expression:str='')->dict:
        """Search a configured English planning corpus, graph first then ordinary.

        Use an English, planning-specific query. Study sites must be read from
        abstracts, never inferred from affiliations. Default publication range
        starts in 2026; set an earlier year explicitly for historical reviews.
        Bibliographic graph links are discovery evidence, not causal evidence.
        Returns complete saved abstracts, DOI provenance and graph paths.
        """
        from datetime import date
        if not 1900<=int(from_year)<=int(to_year or date.today().year)<=date.today().year: raise ValueError('Invalid publication-year range.')
        if not 0<len(query)<=1500: raise ValueError('Use a concise English planning query.')
        q={'id':'mcp_corpus','question':query,'search_query':query,'fts_expression':fts_expression,
            'from_year':from_year,'to_year':to_year}
        limit=max(1,min(limit,25))
        if remote: return remote.retrieve(q,limit,40)
        rag=Retriever(settings['database'],from_year,to_year)
        try: return rag.retrieve(q,limit,40)
        finally: rag.close()

    @server.tool()
    def search_recent_planning_literature(query:str,from_year:int=2026,to_year:int|None=None,limit:int=20,minimum_citations:int=1)->dict:
        """Discover recent planning literature live in OpenAlex, from 2026 by default.

        Translate the user's topic into a concise English query; methods are
        open-ended. Defaults: English Article/Review, valid DOI, at least one
        citation. Explicitly set minimum_citations=0 only when the user asks to
        include uncited new papers. One cached page; not exhaustive. Returned
        metadata may lack abstracts. No live graph expansion or full-text review
        is claimed. Read source evidence before synthesizing a review.
        """
        if remote: return remote.call('/recent',{'query':query,'from_year':from_year,'to_year':to_year,'limit':limit,'minimum_citations':minimum_citations})
        return live.search(query,from_year,to_year,limit,minimum_citations)

    @server.tool()
    def get_planning_paper(doi:str)->dict:
        """Get actual saved/OpenAlex metadata and the available full abstract by DOI.

        Reading this record is abstract-based evidence, not full-text verification.
        Missing abstract, study area or method information must remain unknown.
        """
        doi=normalize_doi(doi)
        if not doi: raise ValueError('A valid DOI is required.')
        if remote: return remote.call('/paper',{'doi':doi})
        if settings.get('database'):
            rag=Retriever(settings['database'])
            try: p=rag.paper(doi)
            finally: rag.close()
            if p: return {'paper':p}
        return {'paper':live.paper(doi)}
    server.run(transport='stdio')

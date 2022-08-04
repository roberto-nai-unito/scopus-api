"""Superclass to access all Scopus retrieval APIs and dump the results."""

from typing import Union

from pybliometrics.scopus.superclasses import Base
from pybliometrics.scopus.utils import get_folder, URLS


class Retrieval(Base):
    def __init__(self,
                 identifier: Union[int, str],
                 api: str,
                 id_type: str = None,
                 **kwds: str
                 ) -> None:
        """Class intended as superclass to perform retrievals.

        :param identifier: The ID to look for.
        :param api: The name of the Scopus API to be accessed.  Allowed values:
                    AbstractRetrieval, AuthorRetrieval, CitationOverview,
                    AffiliationRetrieval.
        :param id_type: The type of the used ID.  Will only take effect for
                        the Abstract Retrieval API.
        :param kwds: Keywords passed on to requests header.  Must contain
                     fields and values specified in the respective
                     API specification.

        Raises
        ------
        KeyError
            If parameter `api` is not one of the allowed values.
        """
        # Construct URL and cache file name
        url = URLS[api]
        if api in ("AbstractRetrieval", "PlumXMetrics"):
            url += id_type + "/"
        if api == 'CitationOverview':
            stem = identifier.replace("/", "")
            if self._citation:
                identifier += "-" + self._citation
        else:
            url += identifier
            stem = identifier.replace('/', '_')
        self._cache_file_path = get_folder(api, self._view)/stem

        # Parse file contents
        params = {'view': self._view, **kwds}
        Base.__init__(self, params=params, url=url, api=api)

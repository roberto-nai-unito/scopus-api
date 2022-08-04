"""Base class object for superclasses."""

from json import dumps, loads
from time import localtime, strftime, time
from typing import Dict, Optional

from pybliometrics.scopus.exception import ScopusQueryError
from pybliometrics.scopus.utils import get_content, SEARCH_MAX_ENTRIES
from tqdm import tqdm


class Base:
    def __init__(self,
                 params: Dict,
                 url: str,
                 api: str,
                 download: bool = True,
                 verbose: bool = False,
                 *args: str, **kwds: str
                 ) -> None:
        """Class intended as base class for superclasses.

        :param params: Dictionary used as header during the API request.
        :param url: The URL to be accessed.
        :param api: The Scopus API to be accessed.
        :param download: Whether to download the query or not.  Has no effect
                         for retrieval requests.
        :param verbose: Whether to print a download progress bar.
        :param args: Keywords passed on `get_content()`
        :param kwds: Keywords passed on `get_content()`

        Raises
        ------
        ValueError
            If `self._refresh` is neither boolean nor numeric.
        """
        # Checks
        try:
            _ = int(self._refresh)
        except ValueError:
            msg = "Parameter refresh needs to be numeric or boolean."
            raise ValueError(msg)

        # Compare age of file to test whether we refresh
        self._refresh, mod_ts = _check_file_age(self)

        # Read or download, possibly with caching
        fname = self._cache_file_path
        search_request = "query" in params
        if fname.exists() and not self._refresh:
            self._mdate = mod_ts
            if search_request:
                self._json = [loads(line) for line in
                              fname.read_text().split("\n") if line]
                self._n = len(self._json)
            else:
                self._json = loads(fname.read_text())
        else:
            resp = get_content(url, api, params, *args, **kwds)
            header = resp.headers
            if search_request:
                # Get number of results
                res = resp.json()
                n = int(res['search-results'].get('opensearch:totalResults', 0))
                self._n = n
                self._json = []
                # Results size check
                if params.get("cursor") is None and n > SEARCH_MAX_ENTRIES:
                    # Stop if there are too many results
                    text = f'Found {n} matches.  The query fails to return '\
                           f'more than {SEARCH_MAX_ENTRIES} entries.  Change '\
                           'your query such that it returns fewer entries.'
                    raise ScopusQueryError(text)
                # Download results page-wise
                if download:
                    data = ""
                    if n:
                        data, header = _parse(res, n, url, api, params,
                                              verbose, *args, **kwds)
                        self._json = data
                else:
                    data = None
            else:
                data = loads(resp.text)
                self._json = data
                data = [data]
            # Set private variables
            self._mdate = time()
            self._header = header
            # Finally write data unless download=False
            if download:
                text = [dumps(item, separators=(',', ':')) for item in data]
                fname.write_text("\n".join(text))

    def get_cache_file_age(self) -> int:
        """Return the age of the cached file in days."""
        diff = time() - self._mdate
        return int(diff / 86400)

    def get_cache_file_mdate(self) -> str:
        """Return the modification date of the cached file."""
        return strftime('%Y-%m-%d %H:%M:%S', localtime(self._mdate))

    def get_key_remaining_quota(self) -> Optional[str]:
        """Return number of remaining requests for the current key and the
        current API (relative on last actual request).
        """
        try:
            return self._header['X-RateLimit-Remaining']
        except AttributeError:
            return None

    def get_key_reset_time(self) -> Optional[str]:
        """Return time when current key is reset (relative on last
        actual request).
        """
        try:
            date = int(self._header['X-RateLimit-Reset'])
            return strftime('%Y-%m-%d %H:%M:%S', localtime(date))
        except AttributeError:
            return None


def _check_file_age(self):
    """Whether a file needs to be refreshed based on its age."""
    refresh = self._refresh
    try:
        mod_ts = self._cache_file_path.stat().st_mtime
        if not isinstance(refresh, bool):
            diff = time() - mod_ts
            days = int(diff / 86400) + 1
            allowed_age = int(self._refresh)
            refresh = allowed_age < days
    except FileNotFoundError:
        refresh = True
        mod_ts = None
    return refresh, mod_ts


def _parse(res, n, url, api, params, verbose, *args, **kwds):
    """Auxiliary function to download results and parse json."""
    cursor = "cursor" in params
    if not cursor:
        start = params["start"]
    _json = res.get('search-results', {}).get('entry', [])
    if verbose:
        # Roundup + 1 for the final iteration
        print(f'Downloading results for query "{params["query"]}":')
        n_chunks = int(n/params['count']) + (n % params['count'] > 0) + 1
        pbar = tqdm(total=n_chunks)
        pbar.update(1)
    # Download the remaining information in chunks
    while n > 0:
        n -= params["count"]
        if cursor:
            pointer = res['search-results']['cursor'].get('@next')
            params.update({'cursor': pointer})
        else:
            start += params["count"]
            params.update({'start': start})
        resp = get_content(url, api, params, *args, **kwds)
        res = resp.json()
        _json.extend(res.get('search-results', {}).get('entry', []))
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    return _json, resp.headers

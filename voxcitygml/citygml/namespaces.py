"""
XML namespace handling and CRS detection for CityGML files.
"""

import logging
import re
from typing import Dict, Optional

try:
    import lxml.etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    HAS_LXML = False
    logging.getLogger(__name__).warning(
        "lxml not installed – falling back to stdlib xml.etree.ElementTree. "
        "Install lxml for faster CityGML parsing: pip install lxml"
    )

log = logging.getLogger(__name__)


DEFAULT_NAMESPACES = {
    'core': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'brid': 'http://www.opengis.net/citygml/bridge/2.0',
    'gml':  'http://www.opengis.net/gml',
    'dem':  'http://www.opengis.net/citygml/relief/2.0',
    'veg':  'http://www.opengis.net/citygml/vegetation/2.0',
    'tran': 'http://www.opengis.net/citygml/transportation/2.0',
    'uro':  'https://www.geospatial.jp/iur/uro/3.0',
}


# -----------------------------------------------------------------------
# Well-known CRS URN → EPSG mapping for German / European CityGML data
# -----------------------------------------------------------------------
_CRS_URN_MAP: Dict[str, str] = {
    # German AdV compound CRS identifiers (horizontal component only)
    'urn:adv:crs:ETRS89_UTM32*DE_DHHN2016_NH': 'EPSG:25832',
    'urn:adv:crs:ETRS89_UTM32*DE_DHHN92_NH':   'EPSG:25832',
    'urn:adv:crs:ETRS89_UTM32':                 'EPSG:25832',
    'urn:adv:crs:ETRS89_UTM33*DE_DHHN2016_NH': 'EPSG:25833',
    'urn:adv:crs:ETRS89_UTM33*DE_DHHN92_NH':   'EPSG:25833',
    'urn:adv:crs:ETRS89_UTM33':                 'EPSG:25833',
    'urn:adv:crs:DE_DHDN_3GK2*DE_DHHN92_NH':   'EPSG:31466',
    'urn:adv:crs:DE_DHDN_3GK3*DE_DHHN92_NH':   'EPSG:31467',
    'urn:adv:crs:DE_DHDN_3GK4*DE_DHHN92_NH':   'EPSG:31468',
    'urn:adv:crs:DE_DHDN_3GK5*DE_DHHN92_NH':   'EPSG:31469',
}


def _resolve_srs_name(srs_name: str) -> Optional[str]:
    """Resolve an srsName string to an EPSG code (or None if geographic).

    Returns
    -------
    str or None
        ``'EPSG:25832'`` etc. for projected CRS, ``None`` for geographic.
    """
    if not srs_name:
        return None

    s = srs_name.strip()

    # Exact match in lookup table
    if s in _CRS_URN_MAP:
        return _CRS_URN_MAP[s]

    # OGC URN / URL EPSG patterns (e.g. EPSG:25832, EPSG::25832, .../EPSG/0/25832)
    m = re.search(r'EPSG(?:::|:|/0/|/)(\d{4,6})', s, re.IGNORECASE)
    if m:
        code = int(m.group(1))
        # Geographic CRS codes – no reprojection needed
        # 4326/4979: WGS84;  6668/6697: JGD2011;  4612: JGD2000
        if code in (4326, 4979, 6668, 6697, 4612):
            return None
        return f'EPSG:{code}'

    # Japanese JGD2011 / JGD2000 geographic (fallback substring check)
    if '6668' in s or '6697' in s or '4612' in s:
        return None

    log.debug("Unknown srsName '%s' – assuming geographic", s)
    return None


def detect_crs_from_root(root) -> Optional[str]:
    """Detect CRS from CityGML XML root's ``gml:Envelope/@srsName``.

    Returns ``'EPSG:25832'`` etc. for projected CRS, ``None`` for geographic.
    """
    gml_ns = 'http://www.opengis.net/gml'
    envelope = root.find(f'{{{gml_ns}}}boundedBy/{{{gml_ns}}}Envelope')
    if envelope is None:
        for path in [
            './/gml:boundedBy/gml:Envelope',
            './/gml:Envelope',
            f'.//{{{gml_ns}}}Envelope',
        ]:
            try:
                envelope = root.find(path, {'gml': gml_ns})
            except Exception:
                pass
            if envelope is not None:
                break
    if envelope is None:
        return None
    return _resolve_srs_name(envelope.get('srsName', ''))


def detect_crs_from_file_header(filepath: str) -> Optional[str]:
    """Quick CRS detection by reading only the first few KB of a GML file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            header = f.read(4096)
        m = re.search(r'srsName="([^"]+)"', header)
        if m:
            return _resolve_srs_name(m.group(1))
    except Exception:
        pass
    return None


def build_namespaces(root) -> Dict[str, str]:
    """Build namespace dictionary from XML root element, with fallbacks.

    Supports CityGML 1.0, 2.0, and 3.0 namespaces.
    """
    nsmap: Dict[str, str] = {}
    default_ns = ''
    if hasattr(root, 'nsmap'):
        # lxml: nsmap may have None key for default namespace
        for k, v in root.nsmap.items():
            if k is not None:
                nsmap[k] = v
            else:
                default_ns = v or ''
    else:
        default_ns = ''

    def pick_ns(prefix: str, keyword: str = None, fallback_key: str = None) -> str:
        uri = nsmap.get(prefix)
        if uri:
            return uri
        if keyword:
            for v in nsmap.values():
                if isinstance(v, str) and keyword in v:
                    return v
            if isinstance(default_ns, str) and keyword in default_ns:
                return default_ns
        return DEFAULT_NAMESPACES.get(fallback_key or prefix, '')

    ns = {
        'core': pick_ns('core', 'citygml/', 'core'),
        'bldg': pick_ns('bldg', 'building', 'bldg'),
        'brid': pick_ns('brid', 'bridge',   'brid'),
        'gml':  pick_ns('gml',  'opengis.net/gml', 'gml'),
        'dem':  pick_ns('dem',  'relief',    'dem'),
        'veg':  pick_ns('veg',  'vegetation','veg'),
        'tran': pick_ns('tran', 'transportation', 'tran'),
        'gen':  pick_ns('gen',  'generics',  'gen'),
        'uro':  pick_ns('uro',  'iur/uro',   'uro'),
        'app':  pick_ns('app',  'appearance', 'app'),
    }

    # If the document default namespace is a CityGML 1.0 URI, use it as 'core'
    if default_ns and 'citygml' in default_ns and ns['core'] != default_ns:
        ns['core'] = default_ns

    return ns

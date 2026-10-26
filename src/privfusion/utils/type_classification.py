import logging
from contextlib import suppress
from typing import Any, cast

import pandas as pd
from risk_assessment.classification.identifiers import (
    SSN,
    City,
    Country,
    CountryCode,
    DateTime,
    DictionaryIdentifier,
    Etnicity,
    Gender,
    Identifier,
    MedicalCode,
    Name,
    NationalIdentity,
    Religion,
    UnitedStateState,
)
from SPARQLWrapper import JSON, SPARQLWrapper


class Continent(DictionaryIdentifier):
    def __init__(self) -> None:
        super().__init__(
            "Continent",
            [
                "Africa",
                "Asia",
                "Europe",
                "North America",
                "Oceania",
                "South America",
            ],
            True,
        )


class ISO3(DictionaryIdentifier):
    def __init__(self) -> None:
        super().__init__(
            "ISOCode3",
            [
                "ABW",
                "AFG",
                "AGO",
                "AIA",
                "ALB",
                "AND",
                "ARE",
                "ARG",
                "ARM",
                "ATG",
                "AUS",
                "AUT",
                "AZE",
                "BDI",
                "BEL",
                "BEN",
                "BES",
                "BFA",
                "BGD",
                "BGR",
                "BHR",
                "BHS",
                "BIH",
                "BLR",
                "BLZ",
                "BMU",
                "BOL",
                "BRA",
                "BRB",
                "BRN",
                "BTN",
                "BWA",
                "CAF",
                "CAN",
                "CHE",
                "CHL",
                "CHN",
                "CIV",
                "CMR",
                "COD",
                "COG",
                "COK",
                "COL",
                "COM",
                "CPV",
                "CRI",
                "CUB",
                "CUW",
                "CYM",
                "CYP",
                "CZE",
                "DEU",
                "DJI",
                "DMA",
                "DNK",
                "DOM",
                "DZA",
                "ECU",
                "EGY",
                "ERI",
                "ESP",
                "EST",
                "ETH",
                "FIN",
                "FJI",
                "FLK",
                "FRA",
                "FRO",
                "FSM",
                "GAB",
                "GBR",
                "GEO",
                "GGY",
                "GHA",
                "GIB",
                "GIN",
                "GMB",
                "GNB",
                "GNQ",
                "GRC",
                "GRD",
                "GRL",
                "GTM",
                "GUY",
                "HKG",
                "HND",
                "HRV",
                "HTI",
                "HUN",
                "IDN",
                "IMN",
                "IND",
                "IRL",
                "IRN",
                "IRQ",
                "ISL",
                "ISR",
                "ITA",
                "JAM",
                "JEY",
                "JOR",
                "JPN",
                "KAZ",
                "KEN",
                "KGZ",
                "KHM",
                "KIR",
                "KNA",
                "KOR",
                "KWT",
                "LAO",
                "LBN",
                "LBR",
                "LBY",
                "LCA",
                "LIE",
                "LKA",
                "LSO",
                "LTU",
                "LUX",
                "LVA",
                "MAC",
                "MAR",
                "MCO",
                "MDA",
                "MDG",
                "MDV",
                "MEX",
                "MHL",
                "MKD",
                "MLI",
                "MLT",
                "MMR",
                "MNE",
                "MNG",
                "MOZ",
                "MRT",
                "MSR",
                "MUS",
                "MWI",
                "MYS",
                "NAM",
                "NCL",
                "NER",
                "NGA",
                "NIC",
                "NIU",
                "NLD",
                "NOR",
                "NPL",
                "NRU",
                "NZL",
                "OMN",
                "OWID_AFR",
                "OWID_ASI",
                "OWID_CYN",
                "OWID_EUN",
                "OWID_EUR",
                "OWID_HIC",
                "OWID_INT",
                "OWID_KOS",
                "OWID_LIC",
                "OWID_LMC",
                "OWID_NAM",
                "OWID_OCE",
                "OWID_SAM",
                "OWID_UMC",
                "OWID_WRL",
                "PAK",
                "PAN",
                "PCN",
                "PER",
                "PHL",
                "PLW",
                "PNG",
                "POL",
                "PRT",
                "PRY",
                "PSE",
                "PYF",
                "QAT",
                "ROU",
                "RUS",
                "RWA",
                "SAU",
                "SDN",
                "SEN",
                "SGP",
                "SHN",
                "SLB",
                "SLE",
                "SLV",
                "SMR",
                "SOM",
                "SPM",
                "SRB",
                "SSD",
                "STP",
                "SUR",
                "SVK",
                "SVN",
                "SWE",
                "SWZ",
                "SXM",
                "SYC",
                "SYR",
                "TCA",
                "TCD",
                "TGO",
                "THA",
                "TJK",
                "TKL",
                "TKM",
                "TLS",
                "TON",
                "TTO",
                "TUN",
                "TUR",
                "TUV",
                "TWN",
                "TZA",
                "UGA",
                "UKR",
                "URY",
                "USA",
                "UZB",
                "VAT",
                "VCT",
                "VEN",
                "VGB",
                "VNM",
                "VUT",
                "WLF",
                "WSM",
                "YEM",
                "ZAF",
                "ZMB",
                "ZWE",
            ],
            False,
        )


class ISOMixed(DictionaryIdentifier):
    def __init__(self) -> None:
        super().__init__(
            "ISOCode2",
            [
                "ID-AC",
                "ID-BA",
                "ID-BB",
                "ID-BE",
                "ID-BT",
                "ID-GO",
                "ID-JA",
                "ID-JB",
                "ID-JI",
                "ID-JK",
                "ID-JT",
                "ID-KB",
                "ID-KI",
                "ID-KR",
                "ID-KS",
                "ID-KT",
                "ID-KU",
                "ID-LA",
                "ID-MA",
                "ID-MU",
                "ID-NB",
                "ID-NT",
                "ID-PA",
                "ID-PB",
                "ID-RI",
                "ID-SA",
                "ID-SB",
                "ID-SG",
                "ID-SN",
                "ID-SR",
                "ID-SS",
                "ID-ST",
                "ID-SU",
                "ID-YO",
                "IDN",
            ],
            False,
        )


class Numeric(Identifier):
    def is_of_this_type(self, text: str) -> bool:
        with suppress(Exception):
            v = float(text)
            if v is not None:
                return True
        return False


logger = logging.getLogger()


identifiers: list[Identifier] = [
    DateTime(),
    # person
    Gender(),
    Religion(),
    SSN(),
    Etnicity(),
    NationalIdentity(),
    Name(),
    # medical
    MedicalCode(),
    # location
    City(),
    Country(),
    UnitedStateState(),
    CountryCode(),
    Continent(),
    ISO3(),
    ISOMixed(),
    # numeric
    Numeric(),
]


identifier_mapping: dict[str, str] = {
    str(identifiers[0]): "http://www.w3.org/2001/XMLSchema#DateTime",
    str(identifiers[1]): "http://dbpedia.org/resource/Gender",
    str(identifiers[2]): "http://dbpedia.org/resource/Religion",
    str(identifiers[3]): "http://dbpedia.org/resource/Social_Security_Number",
    str(identifiers[4]): "https://dbpedia.org/page/Ethnic_group",
    str(identifiers[5]): "https://dbpedia.org/page/Identity_document",
    str(identifiers[6]): "http://xmlns.com/foaf/0.1/Person",
    str(
        identifiers[7],
    ): "https://dbpedia.org/class/yago/WikicatPharmacologicalClassificationSystems",
    str(identifiers[8]): "http://dbpedia.org/resource/City",
    str(identifiers[9]): "https://dbpedia.org/page/Country",
    str(identifiers[10]): "https://dbpedia.org/page/U.S._state",
    str(identifiers[11]): "https://dbpedia.org/page/Country_code",
    str(identifiers[12]): "https://dbpedia.org/page/Continent",
    str(identifiers[13]): "https://dbpedia.org/page/ISO_3166-3",
    str(identifiers[14]): "https://dbpedia.org/page/ISO_3166-2",
    str(identifiers[15]): "http://www.w3.org/2001/XMLSchema#Number",
}


def get_type_mapping(df: pd.DataFrame) -> list[str | None]:
    return [identifier_mapping[t] if t is not None else None for t in extract_types(df)]


def extract_types(df: pd.DataFrame, max_values: int = 5000) -> list[str | None]:
    """Classify each column against the known identifiers.

    Columns with more than ``max_values`` unique values are checked on a
    deterministic random sample of that size: the per-value identifier checks
    are pure Python and dominate runtime on columns with many unique values
    (e.g. floats with thousands of distinct values). The trade-off is that
    very rare values in a column may not be seen.
    """
    d_types: list[str | None] = []

    for col in df.columns:
        data = df[col].unique()
        if len(data) > max_values:
            data = pd.Series(data).sample(n=max_values, random_state=0).to_numpy()

        counters: dict[str, int] = {}

        for point in data:
            s_point = str(point)

            for identifier in identifiers:
                if identifier.is_of_this_type(s_point):
                    counters[str(identifier)] = counters.get(str(identifier), 0) + 1

        if len(counters):
            (best_type, _) = max(counters.items(), key=lambda t: t[1])
            d_types.append(best_type)
        else:
            d_types.append(None)

    return d_types


def find_parents(sparql: SPARQLWrapper, uri: str) -> list[str]:
    query = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?ancestor WHERE {{
      <{uri}> a ?class .
      ?class rdfs:subClassOf+ ?ancestor .
    }}"""
    sparql.setQuery(query)
    try:
        results: dict[str, Any] = cast(dict[str, Any], sparql.queryAndConvert())

        bindings = results["results"]["bindings"]

        return [binding["ancestor"]["value"] for binding in bindings]
    except Exception as e:
        print(f"error querying SPARQL for {uri}")
        print(e)
    return []


def find_common_ancestors(uri1: str, uri2: str) -> list[str]:
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    sparql.setReturnFormat(JSON)

    uri1_parents = find_parents(sparql, uri1)
    uri2_parents = find_parents(sparql, uri2)

    return [common for common in (set(uri1_parents) & (set(uri2_parents)))]


def find_mapping(term: str | None) -> None | list[str]:
    if term is None:
        return None
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    sparql.setReturnFormat(JSON)

    query = (
        """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rds: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?uri
WHERE {
    ?uri rdfs:label \""""
        + term
        + """\"@en
    FILTER (
    !EXISTS {?uri rdf:type rdf:Property}
    )
}"""
    )

    sparql.setQuery(query)

    try:
        results: dict[str, Any] = cast(dict[str, Any], sparql.queryAndConvert())

        # language_terms = [str(result["label"]["value"]) for result in results["results"]["bindings"] if result["label"]["xml:lang"] == language]

        bindings = results["results"]["bindings"]

        return [binding["uri"]["value"] for binding in bindings]

    except Exception as e:
        print(f"error querying sparql for {term}")
        print(e)

    return None

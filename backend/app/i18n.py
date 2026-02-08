"""
Internationalization (i18n) module for Market Wizard.

Supports Polish (PL) and English (EN) languages.
"""

from enum import Enum
from typing import Dict, List


class Language(str, Enum):
    """Supported languages."""
    PL = "pl"
    EN = "en"


# =============================================================================
# ANCHOR STATEMENTS (SSR Engine)
# =============================================================================

DEFAULT_ANCHOR_VARIANT = "paper_general_v4.1b"

ANCHOR_SETS_VARIANTS: Dict[str, Dict[Language, List[Dict[int, str]]]] = {
    "paper_general_v1": {
        Language.PL: [
        {
            1: "Zdecydowanie nie wybiorę tej oferty",
            2: "Raczej nie wybiorę tej oferty",
            3: "Nie jestem pewien, czy wybiorę tę ofertę",
            4: "Raczej wybiorę tę ofertę",
            5: "Zdecydowanie wybiorę tę ofertę",
        },
        {
            1: "Ta oferta w ogóle mnie nie interesuje",
            2: "Jest mało prawdopodobne, że się na to zdecyduję",
            3: "Mogę to rozważyć, ale nie mam pewności",
            4: "Jestem wyraźnie skłonny/a wybrać tę ofertę",
            5: "Na pewno się na to zdecyduję",
        },
        {
            1: "W żadnym wypadku się na to nie zdecyduję",
            2: "Raczej z tego zrezygnuję",
            3: "Mogę wybrać tę opcję, ale równie dobrze mogę nie",
            4: "Jest duża szansa, że wybiorę tę opcję",
            5: "Absolutnie wybiorę tę opcję",
        },
        {
            1: "To zdecydowanie nie jest dla mnie",
            2: "Raczej nie widzę powodu, żeby to wybrać",
            3: "Mam wobec tego neutralne nastawienie",
            4: "To wygląda na coś, co najpewniej wybiorę",
            5: "To dokładnie to, czego szukałem/am",
        },
        {
            1: "Nie mam żadnego zainteresowania tą ofertą",
            2: "Jestem sceptyczny/a wobec tej opcji",
            3: "Może to wybiorę, a może nie",
            4: "Skłaniam się ku tej opcji",
            5: "Bardzo chcę się na to zdecydować",
        },
        {
            1: "To mnie zupełnie nie przekonuje",
            2: "Wolałbym/wolałabym tego nie wybierać",
            3: "Jeszcze nie zdecydowałem/am, czy to wybrać",
            4: "Jest dość prawdopodobne, że to wybiorę",
            5: "Bardzo chętnie to wybiorę",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not choose this option",
            2: "I would probably not choose this option",
            3: "I'm not sure whether I would choose this option",
            4: "I would probably choose this option",
            5: "I would definitely choose this option",
        },
        {
            1: "This offer doesn't interest me at all",
            2: "I'm unlikely to go with this",
            3: "I might consider this, but I'm uncertain",
            4: "I'm clearly inclined to go with this",
            5: "I will certainly go with this",
        },
        {
            1: "There is no way I would choose this",
            2: "I doubt I would choose this",
            3: "I could go either way on this",
            4: "There's a good chance I'll choose this",
            5: "I'm absolutely going to choose this",
        },
        {
            1: "This is not for me at all",
            2: "I don't think this is the right option for me",
            3: "I'm neutral about this option",
            4: "This seems like something I would choose",
            5: "This is exactly what I've been looking for",
        },
        {
            1: "I have zero interest in this offer",
            2: "I'm skeptical about this option",
            3: "Maybe I would choose this, maybe not",
            4: "I'm leaning toward choosing this",
            5: "I'm very eager to choose this",
        },
        {
            1: "This doesn't appeal to me whatsoever",
            2: "I would rather not go with this",
            3: "I haven't decided whether I would choose this",
            4: "I'm fairly likely to choose this",
            5: "I'm very eager to go with this",
        },
        ],
    },
    "paper_general_v2": {
        Language.PL: [
        {
            1: "Na pewno odrzucę tę opcję",
            2: "Raczej odrzucę tę opcję",
            3: "Nie mam jasnej decyzji wobec tej opcji",
            4: "Raczej wybiorę tę opcję",
            5: "Na pewno wybiorę tę opcję",
        },
        {
            1: "To zdecydowanie nie jest coś, na co się zdecyduję",
            2: "Mało prawdopodobne, że się na to zdecyduję",
            3: "Trudno mi powiedzieć, czy się na to zdecyduję",
            4: "Jest duża szansa, że się na to zdecyduję",
            5: "Jestem pewny/a, że się na to zdecyduję",
        },
        {
            1: "Ta oferta zupełnie mi nie odpowiada",
            2: "Ta oferta raczej mi nie odpowiada",
            3: "Ta oferta jest dla mnie neutralna",
            4: "Ta oferta raczej mi odpowiada",
            5: "Ta oferta bardzo mi odpowiada",
        },
        {
            1: "Nie widzę żadnej szansy, żebym to wybrał/a",
            2: "Wątpię, żebym to wybrał/a",
            3: "Mogę to wybrać, ale mogę też nie",
            4: "Prawdopodobnie to wybiorę",
            5: "Zdecydowanie to wybiorę",
        },
        {
            1: "To mnie całkowicie zniechęca",
            2: "To mnie raczej zniechęca",
            3: "To nie budzi we mnie ani chęci, ani niechęci",
            4: "To mnie raczej zachęca",
            5: "To mnie zdecydowanie zachęca",
        },
        {
            1: "Nie jestem skłonny/a wybrać tej opcji",
            2: "Jestem mało skłonny/a wybrać tę opcję",
            3: "Jestem umiarkowanie skłonny/a wybrać tę opcję",
            4: "Jestem dość skłonny/a wybrać tę opcję",
            5: "Jestem bardzo skłonny/a wybrać tę opcję",
        },
        ],
        Language.EN: [
        {
            1: "I would certainly reject this option",
            2: "I would likely reject this option",
            3: "I don't have a clear decision on this option",
            4: "I would likely choose this option",
            5: "I would certainly choose this option",
        },
        {
            1: "There is no chance I would go with this",
            2: "It's unlikely I would go with this",
            3: "I'm undecided about going with this",
            4: "There is a good chance I would go with this",
            5: "I'm sure I would go with this",
        },
        {
            1: "This offer does not fit me at all",
            2: "This offer probably does not fit me",
            3: "This offer feels neutral to me",
            4: "This offer probably fits me",
            5: "This offer fits me very well",
        },
        {
            1: "I see no way I would choose this",
            2: "I doubt I would choose this",
            3: "I could choose this, but I could also pass",
            4: "I will probably choose this",
            5: "I will definitely choose this",
        },
        {
            1: "This strongly discourages me",
            2: "This somewhat discourages me",
            3: "This neither encourages nor discourages me",
            4: "This somewhat encourages me",
            5: "This strongly encourages me",
        },
        {
            1: "I am not inclined to choose this option",
            2: "I am slightly inclined not to choose this option",
            3: "I am moderately inclined either way",
            4: "I am fairly inclined to choose this option",
            5: "I am strongly inclined to choose this option",
        },
        ],
    },
    "paper_general_v3": {
        Language.PL: [
        {
            1: "Definitywnie tego nie wybiorę",
            2: "Raczej tego nie wybiorę",
            3: "Nie jestem zdecydowany/a",
            4: "Raczej to wybiorę",
            5: "Definitywnie to wybiorę",
        },
        {
            1: "To dla mnie zła decyzja",
            2: "To raczej zła decyzja",
            3: "To dla mnie decyzja neutralna",
            4: "To raczej dobra decyzja",
            5: "To dla mnie bardzo dobra decyzja",
        },
        {
            1: "Nie zamierzam się na to decydować",
            2: "Raczej się na to nie zdecyduję",
            3: "Jeszcze nie wiem, co z tym zrobię",
            4: "Raczej się na to zdecyduję",
            5: "Na pewno się na to zdecyduję",
        },
        {
            1: "To zupełnie poza moim wyborem",
            2: "To raczej nie mój wybór",
            3: "To może być mój wybór, ale nie musi",
            4: "To raczej mój wybór",
            5: "To zdecydowanie mój wybór",
        },
        {
            1: "Odrzucam tę opcję bez wahania",
            2: "Skłaniam się do odrzucenia tej opcji",
            3: "Nie mam wyraźnego stanowiska",
            4: "Skłaniam się ku wybraniu tej opcji",
            5: "Wybieram tę opcję bez wahania",
        },
        {
            1: "Ta opcja jest całkowicie nieprzekonująca",
            2: "Ta opcja jest raczej nieprzekonująca",
            3: "Ta opcja jest dla mnie obojętna",
            4: "Ta opcja jest raczej przekonująca",
            5: "Ta opcja jest bardzo przekonująca",
        },
        ],
        Language.EN: [
        {
            1: "I definitely would not choose this",
            2: "I probably would not choose this",
            3: "I am undecided",
            4: "I probably would choose this",
            5: "I definitely would choose this",
        },
        {
            1: "This would be a bad choice for me",
            2: "This is probably a bad choice for me",
            3: "This feels like a neutral choice for me",
            4: "This is probably a good choice for me",
            5: "This would be a very good choice for me",
        },
        {
            1: "I do not plan to go with this",
            2: "I probably will not go with this",
            3: "I still do not know what I would do",
            4: "I probably will go with this",
            5: "I certainly will go with this",
        },
        {
            1: "This is completely outside my choice",
            2: "This is likely not my choice",
            3: "This could be my choice, but not necessarily",
            4: "This is likely my choice",
            5: "This is definitely my choice",
        },
        {
            1: "I reject this option without hesitation",
            2: "I lean toward rejecting this option",
            3: "I have no clear stance",
            4: "I lean toward choosing this option",
            5: "I choose this option without hesitation",
        },
        {
            1: "This option is completely unconvincing to me",
            2: "This option is rather unconvincing to me",
            3: "This option feels indifferent to me",
            4: "This option is rather convincing to me",
            5: "This option is highly convincing to me",
        },
        ],
    },
    "paper_general_v4": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie wiem, czy to kupię",
            4: "Raczej to kupię",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Raczej prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am not sure whether I would buy this",
            4: "I probably would buy this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is somewhat likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Skłaniam się ku zakupowi",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Dość prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I am leaning toward buying this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is fairly likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1a": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Raczej to kupię",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Dość prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I probably would buy this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is fairly likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1b": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Skłaniam się ku zakupowi",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Raczej prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I am leaning toward buying this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is somewhat likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1c": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Skłaniam się ku zakupowi",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Raczej prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Dość prawdopodobne, że to kupię",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I am leaning toward buying this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is somewhat likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "It is fairly likely that I would buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1d": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Skłaniam się ku zakupowi",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Raczej prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja najpewniej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Raczej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I am leaning toward buying this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is somewhat likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is very likely my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would probably decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
    "paper_general_v4.1e": {
        Language.PL: [
        {
            1: "Zdecydowanie tego nie kupię",
            2: "Raczej tego nie kupię",
            3: "Nie mam pewności, czy to kupię",
            4: "Skłaniam się ku zakupowi",
            5: "Zdecydowanie to kupię",
        },
        {
            1: "Bardzo mało prawdopodobne, że to kupię",
            2: "Raczej mało prawdopodobne, że to kupię",
            3: "Trudno powiedzieć, czy to kupię",
            4: "Raczej prawdopodobne, że to kupię",
            5: "Bardzo prawdopodobne, że to kupię",
        },
        {
            1: "Nie zamierzam tego kupować",
            2: "Raczej nie planuję tego kupować",
            3: "Nie mam jeszcze planu zakupu",
            4: "Raczej planuję to kupić",
            5: "Zdecydowanie planuję to kupić",
        },
        {
            1: "Ta opcja jest poza moim wyborem",
            2: "Ta opcja raczej nie jest moim wyborem",
            3: "Ta opcja może być moim wyborem",
            4: "Ta opcja raczej jest moim wyborem",
            5: "Ta opcja zdecydowanie jest moim wyborem",
        },
        {
            1: "To wcale mnie nie przekonuje do zakupu",
            2: "To raczej mnie nie przekonuje do zakupu",
            3: "Mam neutralne nastawienie do zakupu",
            4: "To raczej mnie przekonuje do zakupu",
            5: "To zdecydowanie mnie przekonuje do zakupu",
        },
        {
            1: "Na pewno zrezygnuję z zakupu",
            2: "Raczej zrezygnuję z zakupu",
            3: "Mogę kupić albo zrezygnować",
            4: "Najpewniej zdecyduję się na zakup",
            5: "Na pewno zdecyduję się na zakup",
        },
        ],
        Language.EN: [
        {
            1: "I would definitely not buy this",
            2: "I probably would not buy this",
            3: "I am uncertain whether I would buy this",
            4: "I am leaning toward buying this",
            5: "I would definitely buy this",
        },
        {
            1: "It is very unlikely that I would buy this",
            2: "It is somewhat unlikely that I would buy this",
            3: "It is unclear whether I would buy this",
            4: "It is somewhat likely that I would buy this",
            5: "It is very likely that I would buy this",
        },
        {
            1: "I do not plan to buy this",
            2: "I probably do not plan to buy this",
            3: "I do not have a clear purchase plan yet",
            4: "I probably plan to buy this",
            5: "I definitely plan to buy this",
        },
        {
            1: "This option is outside my choice",
            2: "This option is probably not my choice",
            3: "This option could be my choice",
            4: "This option is probably my choice",
            5: "This option is definitely my choice",
        },
        {
            1: "This does not convince me to buy at all",
            2: "This probably does not convince me to buy",
            3: "I feel neutral about buying this",
            4: "This probably convinces me to buy",
            5: "This definitely convinces me to buy",
        },
        {
            1: "I would definitely pass on buying this",
            2: "I would probably pass on buying this",
            3: "I could buy this or pass",
            4: "I would very likely decide to buy this",
            5: "I would definitely decide to buy this",
        },
        ],
    },
}

# Backward-compatible default constant used by existing call sites.
ANCHOR_SETS: Dict[Language, List[Dict[int, str]]] = ANCHOR_SETS_VARIANTS[DEFAULT_ANCHOR_VARIANT]


# =============================================================================
# PERSONA NAMES AND LOCATIONS
# =============================================================================

FIRST_NAMES: Dict[Language, Dict[str, List[str]]] = {
    Language.PL: {
        "M": [
            "Adam", "Piotr", "Tomasz", "Marcin", "Paweł", "Michał", "Krzysztof",
            "Andrzej", "Jan", "Stanisław", "Jakub", "Mateusz", "Łukasz", "Rafał",
            "Sebastian", "Damian", "Kamil", "Bartosz", "Wojciech", "Grzegorz",
        ],
        "F": [
            "Anna", "Maria", "Katarzyna", "Małgorzata", "Agnieszka", "Barbara",
            "Ewa", "Krystyna", "Magdalena", "Monika", "Joanna", "Aleksandra",
            "Dorota", "Natalia", "Karolina", "Sylwia", "Kinga", "Dominika",
            "Beata", "Justyna",
        ],
    },
    Language.EN: {
        "M": [
            "James", "John", "Michael", "David", "Robert", "William", "Richard",
            "Christopher", "Daniel", "Matthew", "Andrew", "Joseph", "Thomas",
            "Charles", "Steven", "Brian", "Kevin", "Jason", "Mark", "Peter",
        ],
        "F": [
            "Mary", "Patricia", "Jennifer", "Elizabeth", "Linda", "Barbara",
            "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret",
            "Betty", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
        ],
    },
}

LOCATIONS: Dict[Language, Dict[str, List[str]]] = {
    Language.PL: {
        "metropolis": [
            "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań",
        ],
        "large_city": [
            "Gdańsk", "Szczecin", "Bydgoszcz", "Lublin", "Białystok", "Katowice",
            "Gdynia", "Częstochowa", "Radom", "Toruń", "Kielce", "Rzeszów",
        ],
        "medium_city": [
            "Tychy", "Opole", "Gorzów Wielkopolski", "Płock", "Elbląg",
            "Wałbrzych", "Włocławek", "Tarnów", "Chorzów", "Koszalin",
            "Słupsk", "Legnica", "Suwałki", "Jelenia Góra", "Siedlce",
        ],
        "small_city": [
            "Wieliczka", "Piaseczno", "Pruszków", "Ząbki", "Rumia",
            "Zakopane", "Sopot", "Augustów", "Kołobrzeg", "Sandomierz",
            "Ciechanów", "Kwidzyn", "Żywiec", "Nysa", "Bochnia",
        ],
        "rural": [
            "wieś na Mazowszu", "wieś w Małopolsce", "wieś na Podkarpaciu",
            "wieś na Śląsku", "wieś w Wielkopolsce", "wieś na Pomorzu",
            "wieś na Warmii", "wieś na Podlasiu", "wieś w Świętokrzyskiem",
            "wieś na Lubelszczyźnie", "wieś w Łódzkiem", "wieś na Dolnym Śląsku",
        ],
        # Legacy fallback
        "urban": ["Warszawa", "Kraków", "Gdańsk"],
        "suburban": ["Piaseczno", "Sopot", "Wieliczka"],
    },
    Language.EN: {
        "metropolis": [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        ],
        "large_city": [
            "Seattle", "Denver", "Boston", "Nashville", "Portland", "Las Vegas",
        ],
        "medium_city": [
            "Salt Lake City", "Boise", "Tucson", "Fresno", "Spokane",
        ],
        "small_city": [
            "Santa Fe", "Boulder", "Ann Arbor", "Asheville", "Key West",
        ],
        "rural": [
            "rural Texas", "rural Ohio", "rural Iowa", "rural Oregon",
            "rural Alabama", "rural Montana", "rural Vermont",
        ],
        # Legacy fallback
        "urban": ["New York", "Chicago"],
        "suburban": ["Naperville", "Pasadena"],
    },
}

OCCUPATIONS: Dict[Language, List[Dict[str, any]]] = {
    Language.PL: [
        # ISCO 1: Kierownicy (~6% populacji)
        {"name": "menedżer", "min_age": 28, "max_age": 65, "income_min": 7000, "income_max": 25000, "isco": 1},
        {"name": "dyrektor", "min_age": 35, "max_age": 67, "income_min": 12000, "income_max": 40000, "isco": 1},
        {"name": "przedsiębiorca", "min_age": 25, "max_age": 70, "income_min": 5000, "income_max": 50000, "isco": 1},
        
        # ISCO 2: Specjaliści (~22% populacji)
        {"name": "lekarz", "min_age": 26, "max_age": 70, "income_min": 6000, "income_max": 20000, "isco": 2},
        {"name": "dentysta", "min_age": 26, "max_age": 70, "income_min": 7000, "income_max": 25000, "isco": 2},
        {"name": "prawnik", "min_age": 24, "max_age": 70, "income_min": 5000, "income_max": 18000, "isco": 2},
        {"name": "architekt", "min_age": 26, "max_age": 70, "income_min": 5500, "income_max": 15000, "isco": 2},
        {"name": "farmaceuta", "min_age": 25, "max_age": 70, "income_min": 5500, "income_max": 12000, "isco": 2},
        {"name": "programista", "min_age": 22, "max_age": 65, "income_min": 6000, "income_max": 25000, "isco": 2},
        {"name": "inżynier", "min_age": 23, "max_age": 70, "income_min": 5000, "income_max": 15000, "isco": 2},
        {"name": "nauczyciel", "min_age": 23, "max_age": 67, "income_min": 4000, "income_max": 8000, "isco": 2},
        
        # ISCO 3: Technicy i średni personel (~14% populacji)
        {"name": "księgowy", "min_age": 23, "max_age": 70, "income_min": 4000, "income_max": 12000, "isco": 3},
        {"name": "grafik", "min_age": 21, "max_age": 65, "income_min": 3500, "income_max": 12000, "isco": 3},
        {"name": "pielęgniarka", "min_age": 22, "max_age": 67, "income_min": 4500, "income_max": 8000, "isco": 3},
        {"name": "technik", "min_age": 20, "max_age": 65, "income_min": 3500, "income_max": 7000, "isco": 3},
        
        # ISCO 4: Pracownicy biurowi (~8% populacji)
        {"name": "pracownik biurowy", "min_age": 19, "max_age": 67, "income_min": 3500, "income_max": 7000, "isco": 4},
        {"name": "sekretarka", "min_age": 19, "max_age": 60, "income_min": 3000, "income_max": 5500, "isco": 4},
        
        # ISCO 5: Usługi i sprzedaż (~16% populacji)
        {"name": "sprzedawca", "min_age": 18, "max_age": 65, "income_min": 2800, "income_max": 5000, "isco": 5},
        {"name": "fryzjer", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 6000, "isco": 5},
        {"name": "kelner", "min_age": 18, "max_age": 55, "income_min": 2500, "income_max": 4500, "isco": 5},
        {"name": "kucharz", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 7000, "isco": 5},
        {"name": "policjant", "min_age": 21, "max_age": 60, "income_min": 4500, "income_max": 9000, "isco": 5},
        {"name": "strażak", "min_age": 21, "max_age": 55, "income_min": 4500, "income_max": 8000, "isco": 5},
        {"name": "ochroniarz", "min_age": 21, "max_age": 60, "income_min": 3000, "income_max": 5500, "isco": 5},
        
        # ISCO 6: Rolnicy (~8% populacji)
        {"name": "rolnik", "min_age": 18, "max_age": 75, "income_min": 2500, "income_max": 8000, "isco": 6},
        
        # ISCO 7: Robotnicy i rzemieślnicy (~12% populacji)
        {"name": "mechanik", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 7000, "isco": 7},
        {"name": "elektryk", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000, "isco": 7},
        {"name": "pracownik budowlany", "min_age": 18, "max_age": 60, "income_min": 3500, "income_max": 8000, "isco": 7},
        {"name": "stolarz", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 6500, "isco": 7},
        {"name": "spawacz", "min_age": 18, "max_age": 60, "income_min": 4000, "income_max": 9000, "isco": 7},
        
        # ISCO 8: Operatorzy maszyn (~8% populacji)
        {"name": "kierowca", "min_age": 21, "max_age": 67, "income_min": 3500, "income_max": 7000, "isco": 8},
        {"name": "operator produkcji", "min_age": 18, "max_age": 60, "income_min": 3200, "income_max": 5500, "isco": 8},
        
        # ISCO 9: Prace proste (~6% populacji)
        {"name": "magazynier", "min_age": 18, "max_age": 60, "income_min": 3000, "income_max": 5000, "isco": 9},
        {"name": "sprzątaczka", "min_age": 18, "max_age": 65, "income_min": 2800, "income_max": 4000, "isco": 9},
        
        # Statusy specjalne (poza ISCO)
        {"name": "student", "min_age": 18, "max_age": 27, "income_min": 0, "income_max": 2500, "isco": 0},
        {"name": "emeryt", "min_age": 60, "max_age": 100, "income_min": 2000, "income_max": 4500, "isco": 0},
        {"name": "rencista", "min_age": 35, "max_age": 100, "income_min": 1800, "income_max": 3500, "isco": 0},
        {"name": "bezrobotny", "min_age": 18, "max_age": 65, "income_min": 0, "income_max": 1500, "isco": 0},
    ],
    Language.EN: [
        # ISCO 1: Managers (~6% of workforce)
        {"name": "manager", "min_age": 28, "max_age": 65, "income_min": 8000, "income_max": 25000, "isco": 1},
        {"name": "director", "min_age": 35, "max_age": 67, "income_min": 12000, "income_max": 40000, "isco": 1},
        {"name": "entrepreneur", "min_age": 25, "max_age": 70, "income_min": 5000, "income_max": 50000, "isco": 1},
        
        # ISCO 2: Professionals (~22% of workforce)
        {"name": "doctor", "min_age": 26, "max_age": 70, "income_min": 8000, "income_max": 25000, "isco": 2},
        {"name": "dentist", "min_age": 26, "max_age": 70, "income_min": 9000, "income_max": 30000, "isco": 2},
        {"name": "lawyer", "min_age": 24, "max_age": 70, "income_min": 6000, "income_max": 20000, "isco": 2},
        {"name": "architect", "min_age": 26, "max_age": 70, "income_min": 5500, "income_max": 15000, "isco": 2},
        {"name": "pharmacist", "min_age": 25, "max_age": 70, "income_min": 6000, "income_max": 12000, "isco": 2},
        {"name": "software developer", "min_age": 22, "max_age": 65, "income_min": 7000, "income_max": 25000, "isco": 2},
        {"name": "engineer", "min_age": 23, "max_age": 70, "income_min": 5500, "income_max": 15000, "isco": 2},
        {"name": "teacher", "min_age": 23, "max_age": 67, "income_min": 4000, "income_max": 8000, "isco": 2},
        
        # ISCO 3: Technicians (~14% of workforce)
        {"name": "accountant", "min_age": 23, "max_age": 70, "income_min": 4500, "income_max": 12000, "isco": 3},
        {"name": "graphic designer", "min_age": 21, "max_age": 65, "income_min": 4000, "income_max": 12000, "isco": 3},
        {"name": "nurse", "min_age": 22, "max_age": 67, "income_min": 5000, "income_max": 9000, "isco": 3},
        {"name": "technician", "min_age": 20, "max_age": 65, "income_min": 3500, "income_max": 7000, "isco": 3},
        
        # ISCO 4: Clerical workers (~8% of workforce)
        {"name": "office worker", "min_age": 19, "max_age": 67, "income_min": 3500, "income_max": 7000, "isco": 4},
        {"name": "secretary", "min_age": 19, "max_age": 60, "income_min": 3000, "income_max": 5500, "isco": 4},
        
        # ISCO 5: Service and sales (~16% of workforce)
        {"name": "sales associate", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 5000, "isco": 5},
        {"name": "hairdresser", "min_age": 18, "max_age": 65, "income_min": 2500, "income_max": 6000, "isco": 5},
        {"name": "waiter", "min_age": 18, "max_age": 55, "income_min": 2500, "income_max": 5000, "isco": 5},
        {"name": "chef", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000, "isco": 5},
        {"name": "police officer", "min_age": 21, "max_age": 60, "income_min": 5000, "income_max": 10000, "isco": 5},
        {"name": "firefighter", "min_age": 21, "max_age": 55, "income_min": 5000, "income_max": 9000, "isco": 5},
        {"name": "security guard", "min_age": 21, "max_age": 60, "income_min": 3000, "income_max": 5500, "isco": 5},
        
        # ISCO 6: Agricultural workers (~8% of workforce)
        {"name": "farmer", "min_age": 18, "max_age": 75, "income_min": 2500, "income_max": 8000, "isco": 6},
        
        # ISCO 7: Craft workers (~12% of workforce)
        {"name": "mechanic", "min_age": 18, "max_age": 65, "income_min": 3500, "income_max": 8000, "isco": 7},
        {"name": "electrician", "min_age": 18, "max_age": 65, "income_min": 4000, "income_max": 9000, "isco": 7},
        {"name": "construction worker", "min_age": 18, "max_age": 60, "income_min": 3500, "income_max": 8000, "isco": 7},
        {"name": "carpenter", "min_age": 18, "max_age": 65, "income_min": 3000, "income_max": 6500, "isco": 7},
        {"name": "welder", "min_age": 18, "max_age": 60, "income_min": 4000, "income_max": 9000, "isco": 7},
        
        # ISCO 8: Machine operators (~8% of workforce)
        {"name": "driver", "min_age": 21, "max_age": 67, "income_min": 3500, "income_max": 7000, "isco": 8},
        {"name": "production operator", "min_age": 18, "max_age": 60, "income_min": 3200, "income_max": 5500, "isco": 8},
        
        # ISCO 9: Elementary occupations (~6% of workforce)
        {"name": "warehouse worker", "min_age": 18, "max_age": 60, "income_min": 3000, "income_max": 5000, "isco": 9},
        {"name": "cleaner", "min_age": 18, "max_age": 65, "income_min": 2800, "income_max": 4000, "isco": 9},
        
        # Special statuses (outside ISCO)
        {"name": "student", "min_age": 18, "max_age": 27, "income_min": 0, "income_max": 2500, "isco": 0},
        {"name": "retiree", "min_age": 60, "max_age": 100, "income_min": 2000, "income_max": 5000, "isco": 0},
        {"name": "disability pensioner", "min_age": 35, "max_age": 100, "income_min": 1800, "income_max": 4000, "isco": 0},
        {"name": "unemployed", "min_age": 18, "max_age": 65, "income_min": 0, "income_max": 1500, "isco": 0},
    ],
}



# =============================================================================
# LLM PROMPTS
# =============================================================================

def get_persona_prompt(
    language: Language,
    name: str,
    age: int,
    gender: str,
    location: str,
    income: int,
    occupation: str | None,
    product_description: str,
    education: str | None = None,
    marital_status: str | None = None,
    has_children: bool | None = None,
) -> str:
    """
    Build SSR-compliant prompt for synthetic consumer.
    
    Following the methodology from arxiv:2510.08338:
    - Condition LLM on demographic attributes (persona)
    - Ask for textual purchase intent expression
    - Do NOT ask for arguments or reasoning (that biases responses)
    """
    if language == Language.PL:
        gender_word = "kobieta" if gender == "F" else "mężczyzna"
        occupation_line = f"\nPracujesz jako {occupation}." if occupation else ""
        education_line = f" Masz wykształcenie {education}." if education else ""
        
        # Family status
        family_parts = []
        if marital_status:
            family_parts.append(marital_status)
        if has_children is not None:
            family_parts.append("masz dzieci" if has_children else "nie masz dzieci")
        family_line = f"\n{', '.join(family_parts).capitalize()}." if family_parts else ""
        
        return f"""Jesteś {name}, {age}-letni {gender_word} mieszkający w {location}.
Twój miesięczny dochód to około {income} PLN.{occupation_line}{education_line}{family_line}

Rozważ następujący produkt:
{product_description}

Jak bardzo jesteś skłonny/a kupić ten produkt? Odpowiedz naturalnie, tak jak odpowiedziałbyś/odpowiedziałabyś na to pytanie w rozmowie."""

    else:  # EN
        gender_word = "woman" if gender == "F" else "man"
        occupation_line = f"\nYou work as a {occupation}." if occupation else ""
        education_line = f" You have {education} education." if education else ""
        
        # Family status
        family_parts = []
        if marital_status:
            # Translate marital status
            status_map = {
                "kawaler/panna": "single",
                "małżeństwo": "married",
                "rozwiedziony": "divorced",
                "wdowiec/wdowa": "widowed",
            }
            family_parts.append(f"You are {status_map.get(marital_status, marital_status)}")
        if has_children is not None:
            family_parts.append("have children" if has_children else "have no children")
        family_line = f"\n{' and '.join(family_parts)}." if family_parts else ""
        
        return f"""You are {name}, a {age}-year-old {gender_word} living in {location}.
Your monthly income is about ${income}.{occupation_line}{education_line}{family_line}

Consider the following product:
{product_description}

How likely are you to purchase this product? Answer naturally, as you would in a conversation."""


# =============================================================================
# REPORT ANALYSIS PROMPTS
# =============================================================================

def get_report_analysis_prompt(language: Language, payload_json: str) -> str:
    """Build prompt for narrative report analysis from simulation data."""
    if language == Language.PL:
        return (
            "Rola:\n"
            "Jesteś Ekspertem Strategy & Business Intelligence. Twoim zadaniem jest \"wycisniecie\" z raportu SSR "
            "wnioskow, ktorych nie widac na pierwszy rzut oka. Nie streszczaj - interpretuj.\n\n"
            "Zadanie:\n"
            "Przeanalizuj dane i zwroc wynik w formacie JSON.\n\n"
            "Wynik:\n"
            "Zwroc WYLACZNIE poprawny obiekt JSON (bez blokow kodu i bez backtickow). Schemat:\n"
            '{'
            '"narrative":"Glowna analiza (Markdown)",'
            '"agent_summary":"Lista kluczowych faktow (String z lista punktowana)",'
            '"recommendations":"Konkretne dzialania biznesowe (String z lista punktowana)"'
            "}\n\n"
            "Wytyczne do tresci (Instrukcja \"Jak myslec\"):\n\n"
            "1. SEKCJA \"narrative\" (To jest serce raportu, ok. 3000-4500 znakow):\n"
            "   Pisz stylem eseju biznesowego, ale neutralnym i profesjonalnym. "
            "Uzywaj jezyka literalnego i faktow; unikaj metafor, personifikacji i porownan. "
            "Jesli pojawia sie figura stylistyczna, zastap ja opisem zachowania lub wniosku. "
            "Nie tworz nazw segmentow ani etykiet - segmenty opisuj przez cechy "
            "(zawod + dochod + lokalizacja + postawa). "
            "Unikaj pytan retorycznych; uzywaj zdan deklaratywnych. "
            "Kazdy akapit musi zawierac co najmniej jeden element weryfikowalny z danych "
            "(np. liczba, zawod, lokalizacja, dochod). "
            "Podziel na sekcje (uzyj Markdown **Naglowek** i \\n\\n):\n\n"
            "   A. **Psychologia Odbioru i Sentyment:**\n"
            "      - Nie pisz tylko \"jest pozytywnie\". Wyjasnij mechanizmy stojace za ocena "
            "(np. \"fair deal\" lub \"jakosc materialu\" - tylko jesli takie watki rzeczywiscie wystepuja w danych).\n"
            "      - Zanalizuj funkcje produktu: czy to dekoracja, narzedzie spoleczne, prezent, czy cos innego "
            "- ale tylko jesli wynika to z danych.\n"
            "      - Wskaz grupy zawodowe, dla ktorych produkt pelni role \"wentylu bezpieczenstwa\", "
            "jesli takie wskazania sie pojawiaja.\n\n"
            "   B. **Anomalie i Segmentacja:**\n"
            "      - Znajdz sprzecznosci i odchylenia: np. roznice miedzy zawodami estetycznymi a "
            "pragmatycznymi, jesli sa obecne w danych.\n"
            "      - Kto waha sie najbardziej i dlaczego?\n\n"
            "   C. **Strategia i Pozycjonowanie:**\n"
            "      - Zrekonstruuj obecna strategie na podstawie danych (np. jesli pojawiaja sie slowa-klucze "
            "dotyczace jakosci materialu lub funkcji prezentowej).\n"
            "      - Zdefiniuj grupe docelowa psychograficznie (np. dystans do siebie vs \"sztywniacy\"), "
            "jesli mozna to wyczytac z wypowiedzi.\n\n"
            "   D. **Wnioski Koncowe:** Synteza prowadzaca do rekomendacji.\n\n"
            "2. SEKCJA \"agent_summary\" (Konkrety):\n"
            "   - Wypunktuj twarde fakty i powtarzalne wzorce z danych "
            "(np. wzmianki o konkurencji, akceptowane poziomy cen, recurring phrases).\n"
            "   - Nie wymyslaj: jesli jakiegos typu danych brak, pomin.\n\n"
            "3. SEKCJA \"recommendations\" (Strategia):\n"
            "   - Nie dawaj porad typu \"zrob reklamy\". Badz precyzyjny.\n"
            "   - Wymien: potencjal cross-sellingu (co dokupic? zestawy?), sugestie targetowania "
            "(jakie zawody/grupy?), argumenty sprzedazowe (co uwypuklic na Landing Page?).\n"
            "   - Kazda rekomendacja MUSI zakonczyc sie jednym z dopiskow: "
            "\"(wsparte danymi)\" albo \"(sygnal do weryfikacji w szerszej probie)\".\n"
            "   - Uzyj \"(wsparte danymi)\" tylko wtedy, gdy masz wyrazne wsparcie w danych "
            "(np. >=3 niezalezne przyklady). W przeciwnym razie uzyj "
            "\"(sygnal do weryfikacji w szerszej probie)\".\n\n"
            "Zasady techniczne:\n"
            "- Jezyk: Polski.\n"
            "- Formatowanie: Markdown wewnatrz stringow JSON.\n"
            "- Wiernosc: Opieraj sie tylko na dostarczonych danych wejsciowych (JSON SSR).\n\n"
            f"DANE:\n{payload_json}"
        )

    return (
        "Role:\n"
        "You are a Strategy & Business Intelligence Expert. Your task is to \"squeeze\" insights out of the provided "
        "SSR market research data that are not immediately obvious. Do not just summarize - interpret.\n\n"
        "Task:\n"
        "Analyze the input data and return the result in JSON format.\n\n"
        "Output:\n"
        "Return ONLY a valid JSON object (no code blocks, no backticks). Schema:\n"
        '{'
        '"narrative":"Main analysis (Markdown format)",'
        '"agent_summary":"List of key facts (String with a bulleted list)",'
        '"recommendations":"Concrete business actions (String with a bulleted list)"'
        "}\n\n"
        "Content Guidelines (\"How to think\"):\n\n"
        "1. SECTION \"narrative\" (The heart of the report, approx. 3000-4500 characters):\n"
        "   Write in a business-essay style, but neutral and professional. "
        "Use literal language and facts; avoid metaphors, personification, and comparisons. "
        "If a figure of speech appears, replace it with a literal behavioral description or conclusion. "
        "Do not create segment names or labels - describe segments by attributes "
        "(occupation + income + location + attitude). "
        "Avoid rhetorical questions; use declarative sentences. "
        "Each paragraph must include at least one verifiable data element "
        "(e.g., number, occupation, location, income). "
        "Divide into sections (use Markdown **Header** and \\n\\n):\n\n"
        "   A. **Psychology of Perception & Sentiment:**\n"
        "      - Don't just say \"positive.\" Explain the mechanisms behind the evaluation "
        "(e.g., \"fair deal\" or \"material quality\" - only if such themes actually appear in the data).\n"
        "      - Analyze the product function: decor vs social tool vs gift - but only if supported by the responses.\n"
        "      - Identify professional groups for whom the product acts as a \"safety valve,\" if such signals are present.\n\n"
        "   B. **Anomalies & Segmentation:**\n"
        "      - Find contradictions and deviations: e.g., differences between design-sensitive vs pragmatic professions, "
        "if present in the data.\n"
        "      - Who hesitates the most and why?\n\n"
        "   C. **Strategy & Positioning:**\n"
        "      - Reconstruct the current strategy from the data (e.g., if keyword signals about material quality "
        "or gift value appear).\n"
        "      - Define the target audience psychographically (e.g., self-distance vs \"stiff\" corporate types), "
        "only if evidenced in responses.\n\n"
        "   D. **Strategic Synthesis:** Summary leading into recommendations.\n\n"
        "2. SECTION \"agent_summary\" (Hard Facts):\n"
        "   - Bullet point hard facts and recurring patterns from the data "
        "(e.g., competitor mentions, acceptable price levels, recurring phrases).\n"
        "   - Do not invent; if a data type is missing, omit it.\n\n"
        "3. SECTION \"recommendations\" (Strategy):\n"
        "   - Do not give generic advice like \"run ads.\" Be precise.\n"
        "   - List: cross-selling potential (what to bundle?), targeting suggestions (which professions/groups?), "
        "sales arguments (what to highlight on the landing page?).\n"
        "   - Every recommendation MUST end with one of these suffixes: "
        "\"(supported by data)\" or \"(signal to validate with a broader sample)\".\n"
        "   - Use \"(supported by data)\" only when there is clear support in the data "
        "(e.g., >=3 independent examples). Otherwise use "
        "\"(signal to validate with a broader sample)\".\n\n"
        "Technical Rules:\n"
        "- Language: English.\n"
        "- Formatting: Markdown inside JSON strings.\n"
        "- Fidelity: Use only the provided SSR input data (JSON).\n\n"
        f"DATA:\n{payload_json}"
    )


def get_report_analysis_sanitize_prompt(language: Language, analysis_json: str) -> str:
    """Build prompt to sanitize report analysis into a literal, neutral style."""
    if language == Language.PL:
        return (
            "Zadanie: Oczyść styl tekstu analizy bez zmiany faktów.\n"
            "Wejście: JSON z polami narrative, agent_summary, recommendations.\n"
            "Wyjście: ZWRÓĆ WYŁĄCZNIE poprawny JSON o tym samym schemacie.\n\n"
            "Zasady:\n"
            "- Nie dodawaj nowych informacji ani wniosków.\n"
            "- Nie usuwaj faktów, liczb, zawodów, lokalizacji ani cytowanych przykładów.\n"
            "- Zamień metafory, personifikacje i storytelling na opis literalny.\n"
            "- Nie używaj etykiet dla grup (np. \"segment X\"). Opisuj je przez cechy.\n"
            "- Unikaj pytań retorycznych. Używaj zdań deklaratywnych.\n"
            "- Zachowaj strukturę akapitów i list.\n"
            "- Nie zmieniaj ani nie usuwaj suffixów w rekomendacjach "
            "(np. \"(wsparte danymi)\" / \"(sygnal do weryfikacji w szerszej probie)\").\n\n"
            f"JSON:\n{analysis_json}"
        )
    return (
        "Task: Sanitize the analysis style without changing facts.\n"
        "Input: JSON with fields narrative, agent_summary, recommendations.\n"
        "Output: RETURN ONLY valid JSON with the same schema.\n\n"
        "Rules:\n"
        "- Do not add new information or conclusions.\n"
        "- Do not remove facts, numbers, occupations, locations, or cited examples.\n"
        "- Replace metaphors, personification, and storytelling with literal descriptions.\n"
        "- Replace playful labels or nicknames (e.g., \"Neighbor War\", \"Office Prank\") with neutral descriptions.\n"
        "- Do not use group labels (e.g., \"segment X\"). Describe by attributes.\n"
        "- Avoid rhetorical questions. Use declarative sentences.\n"
        "- Preserve paragraph and list structure.\n"
        "- Do not change or remove recommendation suffixes "
        "(e.g., \"(supported by data)\" / \"(signal to validate with a broader sample)\").\n\n"
        f"JSON:\n{analysis_json}"
    )

# =============================================================================
# UI LABELS
# =============================================================================

UI_LABELS: Dict[Language, Dict[str, str]] = {
    Language.PL: {
        "app_title": "🔮 Market Wizard",
        "app_subtitle": "Analizator Rynku oparty na metodologii SSR",
        "tab_simulation": "📊 Symulacja Podstawowa",
        "tab_ab_test": "🔬 Test A/B",
        "tab_price": "💰 Analiza Cenowa",
        "tab_about": "ℹ️ O metodologii",
        "product_label": "Opis produktu",
        "product_placeholder": "Np. Pasta do zębów z węglem aktywnym, 75ml, cena 24.99 PLN",
        "target_group": "Grupa docelowa",
        "age_min": "Wiek min",
        "age_max": "Wiek max",
        "gender": "Płeć",
        "gender_all": "Wszystkie",
        "income": "Dochód",
        "income_all": "Wszystkie",
        "income_low": "Niski",
        "income_medium": "Średni",
        "income_high": "Wysoki",
        "location": "Lokalizacja",
        "location_all": "Wszystkie",
        "location_urban": "Miasto",
        "location_suburban": "Przedmieścia",
        "location_rural": "Wieś",
        "n_agents": "Liczba agentów",
        "run_simulation": "🚀 Uruchom symulację",
        "run_ab_test": "🔬 Uruchom test A/B",
        "run_price_analysis": "💰 Analizuj wrażliwość cenową",
        "results_title": "📊 Wyniki Symulacji",
        "mean_purchase_intent": "Średnia intencja zakupu",
        "n_agents_result": "Liczba agentów",
        "distribution": "Rozkład odpowiedzi",
        "scale_1": "Zdecydowanie NIE",
        "scale_2": "Raczej nie",
        "scale_3": "Ani tak, ani nie",
        "scale_4": "Raczej tak",
        "scale_5": "Zdecydowanie TAK",
        "opinions_title": "📝 Przykładowe opinie agentów",
        "variant_a": "Wariant A",
        "variant_b": "Wariant B",
        "price_min": "Cena min (PLN)",
        "price_max": "Cena max (PLN)",
        "price_points": "Punkty cenowe",
        "demand_curve": "Krzywa popytu",
        "optimal_price": "Optymalna cena",
        "elasticity": "Elastyczność cenowa",
        "winner": "Zwycięzca",
        "lift": "Lift",
        "error_no_product": "❌ Wprowadź opis produktu",
        "error_no_variants": "❌ Wprowadź opisy obu wariantów",
        "success": "✅ Symulacja zakończona pomyślnie",
        "extract_url": "🔗 Pobierz z URL",
    },
    Language.EN: {
        "app_title": "🔮 Market Wizard",
        "app_subtitle": "Market Analyzer based on SSR methodology",
        "tab_simulation": "📊 Basic Simulation",
        "tab_ab_test": "🔬 A/B Test",
        "tab_price": "💰 Price Analysis",
        "tab_about": "ℹ️ About",
        "product_label": "Product description",
        "product_placeholder": "E.g. Activated charcoal toothpaste, 75ml, price $9.99",
        "target_group": "Target audience",
        "age_min": "Age min",
        "age_max": "Age max",
        "gender": "Gender",
        "gender_all": "All",
        "income": "Income",
        "income_all": "All",
        "income_low": "Low",
        "income_medium": "Medium",
        "income_high": "High",
        "location": "Location",
        "location_all": "All",
        "location_urban": "Urban",
        "location_suburban": "Suburban",
        "location_rural": "Rural",
        "n_agents": "Number of agents",
        "run_simulation": "🚀 Run simulation",
        "run_ab_test": "🔬 Run A/B test",
        "run_price_analysis": "💰 Analyze price sensitivity",
        "results_title": "📊 Simulation Results",
        "mean_purchase_intent": "Mean purchase intent",
        "n_agents_result": "Number of agents",
        "distribution": "Response distribution",
        "scale_1": "Definitely NOT",
        "scale_2": "Probably not",
        "scale_3": "Neutral",
        "scale_4": "Probably yes",
        "scale_5": "Definitely YES",
        "opinions_title": "📝 Sample agent opinions",
        "variant_a": "Variant A",
        "variant_b": "Variant B",
        "price_min": "Price min ($)",
        "price_max": "Price max ($)",
        "price_points": "Price points",
        "demand_curve": "Demand curve",
        "optimal_price": "Optimal price",
        "elasticity": "Price elasticity",
        "winner": "Winner",
        "lift": "Lift",
        "error_no_product": "❌ Please enter a product description",
        "error_no_variants": "❌ Please enter descriptions for both variants",
        "success": "✅ Simulation completed successfully",
        "extract_url": "🔗 Fetch from URL",
    },
}


def get_label(language: Language, key: str) -> str:
    """Get UI label for given language and key."""
    return UI_LABELS.get(language, UI_LABELS[Language.EN]).get(key, key)


def get_anchor_sets(language: Language, variant: str | None = None) -> List[Dict[int, str]]:
    """Get anchor statements for given language and optional variant."""
    key = (variant or DEFAULT_ANCHOR_VARIANT).strip().lower()
    variant_sets = ANCHOR_SETS_VARIANTS.get(key) or ANCHOR_SETS_VARIANTS[DEFAULT_ANCHOR_VARIANT]
    return variant_sets.get(language, variant_sets[Language.EN])


def get_anchor_variants() -> List[str]:
    """Return available anchor variant keys."""
    return sorted(ANCHOR_SETS_VARIANTS.keys())

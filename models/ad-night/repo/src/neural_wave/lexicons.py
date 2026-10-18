"""Lexicons: built public resources (lazy, cached) plus small curated lists.

Built resources live in `neural_wave/data/*.gz` (see `scripts/build_lexicons.py`). Curated lists
below come from the dataset analysis (`outputs/eda/DATASET_ANALYSIS.md`), especially the trap
catalogue: scribe products, eponyms, title stop-words, collision words.
"""

from __future__ import annotations

import gzip
import re
from functools import cache
from importlib import resources


def _rows(name: str) -> list[list[str]]:
    with resources.files("neural_wave.data").joinpath(name).open("rb") as raw:
        with gzip.open(raw, "rt", encoding="utf-8") as fh:
            return [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]


def _csv(s: str) -> frozenset[str]:
    """Comma-separated list; entries may wrap across lines."""
    return frozenset(re.sub(r"\s+", " ", w).strip().lower() for w in s.split(",") if w.strip())


# --------------------------------------------------------------------------- built resources


@cache
def first_names() -> dict[str, float]:
    """First name -> popularity score (sum of yearly % since 1925)."""
    return {r[0]: float(r[1]) for r in _rows("first_names.tsv.gz")}


@cache
def surnames() -> dict[str, int]:
    """US Census 2010 surnames with count >= 800."""
    return {r[0]: int(r[1]) for r in _rows("surnames.tsv.gz")}


@cache
def wordfreq() -> dict[str, tuple[int, int]]:
    """word -> (lowercase uses, mid-sentence capitalised uses) in a cased medical corpus."""
    return {r[0]: (int(r[1]), int(r[2])) for r in _rows("wordfreq_medconv.tsv.gz")}


@cache
def icd10cm() -> dict[str, str]:
    """ICD-10-CM 2026 code (dotted) -> long description (headers and billable codes)."""
    return {r[0]: r[2] for r in _rows("icd10cm_2026.tsv.gz")}


@cache
def icd_index_paths() -> dict[str, list[str]]:
    """Code -> alphabetic-index paths that lead to it (lay synonyms, e.g. L21.0 <- 'Dandruff')."""
    out: dict[str, list[str]] = {}
    for code, _kind, path in _rows("icd_index_paths.tsv.gz"):
        out.setdefault(code.rstrip("-").rstrip("."), []).append(path)
    return out


@cache
def icd_main_terms() -> frozenset[str]:
    return frozenset(r[0] for r in _rows("icd_index_main_terms.txt.gz"))


@cache
def drug_names() -> frozenset[str]:
    return frozenset(r[0] for r in _rows("drugs.txt.gz"))


@cache
def drug_tokens() -> frozenset[str]:
    """Single-word drug names that are not English dictionary words, names or frequent words."""
    wf = wordfreq()
    out = set()
    for (t,) in _rows("drug_tokens.txt.gz"):
        if wf.get(t, (0, 0))[0] < 400 and t not in first_names() and surnames().get(t, 0) < 1500:
            out.add(t)
    return frozenset(out | DRUG_EXTRA)


@cache
def places() -> dict[str, tuple[str, int, str]]:
    """Place name -> (type city/state/country, population, country code)."""
    out: dict[str, tuple[str, int, str]] = {}
    for name, typ, pop, cc in _rows("places.tsv.gz"):
        prev = out.get(name)
        if prev is None or (prev[0] == "city" and typ != "city") or (typ == prev[0] and int(pop) > prev[1]):
            out[name] = (typ, int(pop), cc)
    return out


@cache
def medical_vocab() -> frozenset[str]:
    """Single words that occur in ICD-10-CM descriptions or index main terms (lowercase)."""
    words: set[str] = set()
    for d in icd10cm().values():
        words.update(re.findall(r"[a-z]{3,}", d.lower()))
    for t in icd_main_terms():
        words.update(re.findall(r"[a-z]{3,}", t))
    return frozenset(words)


# --------------------------------------------------------------------------- name helpers


def lower_uses(w: str) -> int:
    return wordfreq().get(w.lower(), (0, 0))[0]


def is_common_word(w: str, threshold: int = 30) -> bool:
    """True if `w` is regularly used as an ordinary lowercase word (`will`, `white`, `cook`)."""
    w = w.lower()
    return w in COMMON_EXTRA or lower_uses(w) >= threshold


def is_ordinary_word(w: str, threshold: int = 30) -> bool:
    """An ordinary lowercase word (`mobile`, `reading`, `turkey`), not a proper noun that people sometimes
    type in lowercase (`hyderabad`, `mumbai`): mostly used lowercase in a cased corpus."""
    w = w.lower()
    low, cap = wordfreq().get(w, (0, 0))
    return (low >= threshold and low > 3 * max(cap, 1)) or (w in COMMON_EXTRA and w not in places())


def name_score(w: str) -> float:
    """Heuristic 0..1 that a lowercase token is a person name in clinical text."""
    w = w.lower().removesuffix("'s")
    if w in NAME_STOP or w in TITLES or len(w) < 2:
        return 0.0
    fn, sn = first_names().get(w, 0.0), surnames().get(w, 0)
    if not fn and not sn:
        return 0.0
    low, cap = wordfreq().get(w, (0, 0))
    base = 0.95 if fn >= 0.05 or sn >= 20000 else 0.8 if fn or sn >= 3000 else 0.6
    if low >= 30 or w in COMMON_EXTRA:
        base *= 0.3
    elif low > 3 * max(cap, 1):
        base *= 0.6
    if w in drug_tokens():
        base *= 0.3
    return base


# --------------------------------------------------------------------------- curated lists

TITLES = _csv("mr, mrs, ms, miss, mister, missus, dr, doctor, sir, madam")
PATIENT_TITLES = _csv("mr, mrs, ms, miss, mister, missus")

# Tokens that never are person names here (EDA trap catalogue + frequent false positives).
NAME_STOP = _csv(
    """dax, dragon, nuance, epic, cerner, siri, alexa, the, a, an, and, or, but, so, um, uh, okay, ok, yeah, yes,
    no, hi, hello, hey, there, everyone, everybody, guys, doc, doctor, document, again, how, what, i, i'm, it's,
    this, that, you, well, good, nice, oh, all, back, to, my, we, mom, dad, mama, papa, buddy, sweetie, honey, dear,
    sir, ma'am, folks, little, big, thank, thanks, sorry, come, let's, let, just, is, are, was, were, do, does, did,
    have, has, had, can, could, would, should, might, will, in, on, at, for, of, with, up, down, out, now, then,
    here, home, right, left, over, about, like, look, see, say, know, think, feel, morning, afternoon, evening,
    today, tomorrow, yesterday, patient, patients, nurse, mrs, mr, ms, miss, mister, dr, pa, np, md, rn, sure,
    alright, great, perfect, awesome, wonderful, excellent, man, kiddo, sweetheart, lady, pal, friend,
    everything, anything, nothing, something, one, two, first, last, next, team, folks, ladies, gentlemen,
    january, february, march, april, may, june, july, august, september, october, november, december,
    monday, tuesday, wednesday, thursday, friday, saturday, sunday, christmas, easter, thanksgiving,
    ray, x, xray, pe, mri, ct, ekg, ecg, er, ed, icu, iv, gi, ent, ob, gyn, pt, ot, covid, tylenol, advil,
    motrin, aleve, ibuprofen, lisinopril, metformin, medicare, medicaid, kaiser, cvs, walgreens, walmart,
    god, lord, jesus, christ, heaven, hell, damn, gosh, geez, jeez, wow, huh, hmm, mm, mhm, ah, aw, ugh,
    he, she, him, his, her, hers, they, them, their, theirs, we, us, our, me, my, mine, you, your, yours, it, its,
    who, whom, whose, which, what, someone, somebody, anyone, anybody, nobody, people, person, guy, gal, sir,
    male, female, man, woman, boy, girl, gentleman, lady, child, baby, son, daughter, mother, father, wife,
    husband, pleasant, very, old, year, years, month, months"""
)

# Words used every day in clinical dialogue that also appear in the name lists (EDA §4.5).
COMMON_EXTRA = _csv(
    """will, may, mark, bill, hope, grace, faith, joy, summer, winter, spring, fall, april, june, august,
    candy, sunny, dawn, rich, nick, ray, pat, sue, rob, chase, guy, frank, art, gene, don, max, jean, lee,
    white, black, brown, green, gray, young, long, short, hill, wood, rose, bell, king, price, cook, walker,
    baker, hunter, fisher, miller, taylor, carter, mason, porter, turner, cole, ward, page, lane, wells,
    banks, fields, rivers, waters, snow, frost, stone, rock, west, north, south, east, love, hand, foot,
    head, back, side, best, small, little, major, rice, bush, root, branch, hurt, holiday, pain, mann, sharp,
    strong, hardy, gross, ball, crane, fox, wolf, lamb, bird, swan, drake, roach, lyme, colt, english, french,
    german, irish, christian, tuesday, monday, early, late, sterling, story, case"""
)

MEDICAL_EPONYMS = _csv(
    """alzheimer, parkinson, crohn, hodgkin, graves, paget, cushing, addison, lyme, bell, wilson, huntington,
    gehrig, hashimoto, sjogren, raynaud, barrett, hirschsprung, guillain, barre, marfan, ehlers, danlos, down,
    turner, klinefelter, kawasaki, still, achilles, bartholin, baker, dupuytren, osgood, schlatter, quervain,
    tourette, asperger, wegener, reiter, whipple, meniere, colles, lisfranc, jones, morton, bennett, hill, sachs,
    kaposi, burkitt, wilms, ewing, mallory, weiss, zollinger, ellison, cheyne, stokes, babinski, homan, homans,
    murphy, rovsing, mcburney, hawkins, kennedy, neer, lachman, mcmurray, phalen, tinel, finkelstein, spurling,
    ober, thomas, thompson, apley, allen, romberg, glasgow, apgar, braden, morse, epley, dix, hallpike, foley,
    kegel, heimlich, pap, kirschner, salter, harris, gleason, bishop, ortolani, barlow, charcot, holter,
    doppler, lasik, fowler, trendelenburg, sims, hoffman, hoffmann, smith, magenis, brugada, wolff, white,
    parkinson's, valsalva, mcburney's, schober, faber, patrick, gaenslen, hawkins-kennedy, yergason, speed,
    jobe, adson, roos, wright, lhermitte, kernig, brudzinski, chvostek, trousseau, cullen, grey-turner,
    kehr, levine, austin, flint, graham, steell, kussmaul, korotkoff, lown, wenckebach, mobitz, bundle, purkinje,
    raynaud's, cushing's, addison's, crohn's, hodgkin's, graves', bell's, wilson's, huntington's, sjogren's,
    hashimoto's, barrett's, paget's, baker's, dupuytren's, tourette's, meniere's, murphy's, tinel's,
    phalen's, homan's, spurling's, mcmurray's, lachman's, neer's, finkelstein's, colles', crigler, najjar,
    gilbert, peyronie, osler, weber, rendu, sturge, kawasaki's, stevens, johnson, dressler, zenker, schatzki,
    epstein, barr, coxsackie, norwalk, legionella, listeria, salmonella, shigella, ebstein, fallot, eisenmenger,
    kaposi's, lou, tommy, john, pitt, mallet, boutonniere, jersey, golfer's, tennis, swimmer's, athlete's,
    jock, runner's, jumper's, housemaid's, nursemaid's, student's, miner's, farmer's, smoker's, writer's, horner,
    horner's, pancoast, reed, sternberg, flint, montreal, roth, janeway, osler's, virchow, virchow's, sister mary joseph,
    trousseau's, chvostek's, brudzinski's, kernig's, rinne, rinne's, weber's, dix-hallpike, glasgow, apgar, gleason,
    breslow, clark, ranson, child-pugh, wells, centor, curb, nihss, killip, nyha, gold, mallampati, asa, monro,
    munro, fitzpatrick, willis, vater, oddi, langerhans, auerbach, meissner, brunner, peyer, purkinje, wernicke,
    broca, korsakoff, lewy, kaposi, sjogren, bowman, henle, hunter, alcock, calot, scarpa, poupart"""
)
# Context that turns an eponym into a disease/test/sign: `lyme titer`, `murphy's sign`, `thompson test`.
EPONYM_NEXT = (
    r"(?:'s|s'|’s)?\s+(?:disease|syndrome|sign|test|tests|maneuver|manoeuvre|fracture|palsy|titer|titers|"
    r"cyst|contracture|tendon|tendonitis|tendinitis|tendinopathy|thyroiditis|esophagus|ulcer|phenomenon|"
    r"lymphoma|sarcoma|tumor|tumour|dementia|chorea|neuroma|criteria|score|scale|classification|"
    r"position|reflex|procedure|operation|repair|catheter|wire|test|exam|examination|monitor|orbitopathy|ophthalmopathy|"
    r"eye disease|murmur|murmurs|ulcers|nodules|spots|staging|stage|maneuvre|skin type|phototype|bodies|"
    r"positive|negative|elbow|knee|toe|foot|shoulder|ear|itch|lung|cramp|arthritis|infection|"
    r"block|type|grade|stage|class|lesion|node|nodes|body|bodies|cell|cells|virus|fever|bacteria|"
    r"distress|triad|pentad|ratio|index|fluid|sinus|deformity|pain|injury|tear)\b"
)

SCRIBE_PRODUCTS = _csv("dax, dragon, nuance, epic, cerner, siri, alexa, mychart, my chart, golive, go-live")
# Organisations named for clinical reasons, not about the patient (guideline bodies, agencies, programs).
NON_PERSONAL_ORGS = _csv(
    """national comprehensive cancer network, nccn, cdc, fda, who, world health organization, american heart
    association, aha, american diabetes association, ada, uspstf, acc, american college of cardiology, nih,
    american cancer society, acs, cms, aap, acog, gold, idsa, kdigo, nice, medicare, medicaid"""
)
GENERIC_PLACES = _csv(
    """basement, basement stairs, backyard, back yard, front yard, yard, garage, kitchen, bathroom, bedroom, stairs,
    staircase, porch, deck, driveway, sidewalk, curb, street, road, highway, trail, trails, bike trail, hiking trails,
    new trails, park, gym, pool, field, court, league, local league, beach, lake, woods, mountains, store, mall, church,
    school, office, home, house, apartment, work, job, farm, barn, ed, er, hospital, clinic, pharmacy, nursing home,
    rehab, urgent care, the y, cul-de-sac, culdesac, neighborhood, downtown, city, town, country, state, abroad"""
)

RELATIONS = _csv(
    """wife, husband, spouse, partner, boyfriend, girlfriend, fiance, fiancee, mom, mother, mommy, mama, dad,
    father, daddy, papa, son, daughter, kid, child, baby, brother, sister, sibling, grandmother, grandma,
    granny, grandfather, grandpa, granddaughter, grandson, aunt, uncle, cousin, nephew, niece, stepmother,
    stepfather, stepson, stepdaughter, stepdad, stepmom, roommate, neighbor, friend, buddy, coworker, colleague,
    boss, caregiver, twin, godmother, godfather, oldest, youngest, eldest"""
)

STAFF_WORDS = _csv(
    """nurse, assistant, medical assistant, ma, receptionist, scheduler, therapist, physical therapist, pt,
    pharmacist, dietitian, nutritionist, pa, np, nurse practitioner, physician assistant, tech, technician,
    coordinator, social worker, case manager, secretary, front desk, partner, colleague, resident, fellow,
    student, intern, scribe, surgeon, specialist, cardiologist, dermatologist, neurologist, oncologist,
    radiologist, urologist, endocrinologist, gastroenterologist, rheumatologist, podiatrist, chiropractor,
    psychiatrist, psychologist, counselor, provider, physician, pcp, colleague"""
)

# After `mr`, `dr`, `doctor`: words that are not names (`mr arthrogram`, `doctor said`).
TITLE_STOP = _csv(
    """arthrogram, angiogram, angiography, enterography, imaging, scan, scans, mri, mra, venogram, put, said,
    told, is, was, will, who, and, has, wants, thinks, that, to, of, for, about, can, did, if, here, says, does,
    would, should, could, or, in, on, at, with, may, might, also, just, then, there, they, you, i, we, he, she,
    it, a, an, the, my, your, his, her, their, our, visit, appointment, recommended, mentioned, wanted, thought,
    called, ordered, prescribed, referred, gave, sent, wrote, looked, checked, saw, sees, see, seen, asked, know,
    think, feel, believe, yet, already, still, today, last, next, first, right, okay, yeah, um, uh, so, when,
    what, how, where, why, because, but, since, before, after, once, while, not, no, yes, please, sorry, let,
    look, looks, do, done, doing, going, comes, come, came, get, gets, got, give, had, have, be, been, am, are,
    were, us, them, me, him, this, any, some, every, all, both, more, most, much, many, few, other, another,
    very, too, really, pretty, well, good, new, old, little, day, time, week, month, year, again, back, up,
    down, out, over, like, from, by, nurse, office, as, than, visits, want, needs, need, say, tells, tell, ask,
    asks, recommend, recommends, order, orders, write, writes, call, calls, which, whose, whom, either,
    neither, actually, basically, probably, maybe, definitely, certainly, obviously, apparently, hopefully,
    office, offices, appointments, note, notes, team, dax, dragon, kind, type, sort, lady, man, guy, person,
    one, ones, friend, friends, who's, that's, i'll, i'd, i've, you're, he's, she's, we'll, they're, gave,
    would've, could've, removed, did, fixed, checked, read, reading, reviewed, found, shows, showed, and,
    hi, hello, hey, smith's"""
) - frozenset()

# Greeting / vocative cue words that may precede a bare name in dialogue.
GREETINGS = r"(?:hi|hello|hey|hiya|howdy|good morning|good afternoon|good evening|morning|afternoon|evening|welcome|welcome back|nice to meet you|nice to see you|good to see you|great to see you)"
VOCATIVE_PRE = r"(?:so|okay|ok|alright|all right|well|thanks|thank you|bye|goodbye|see you|right|yes|yeah|now|great|good|perfect|sure|and|oh|hey|hi|hello|listen|look|um|uh|sorry)"

MONTHS = r"(?:january|february|march|april|may|june|july|august|september|sept|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
MONTHS_FULL = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"

INSURERS = _csv(
    """medicare, medicaid, tricare, kaiser, kaiser permanente, aetna, cigna, humana, anthem, bluecross,
    blue cross, blue shield, blue cross blue shield, unitedhealthcare, united healthcare, united health,
    wellcare, molina, centene, ambetter, champva, star health, star health and allied insurance,
    hdfc ergo, icici lombard, bajaj allianz, max bupa, niva bupa, care health, religare, new india assurance,
    oriental insurance, united india insurance, national insurance, ayushman bharat, cghs, esic"""
)
SPORTS_TEAMS = _csv(
    """yankees, red sox, dodgers, cubs, braves, astros, phillies, mets, brewers, orioles, giants, padres,
    mariners, rangers, royals, twins, tigers, guardians, indians, reds, pirates, cardinals, rockies,
    diamondbacks, marlins, nationals, angels, athletics, white sox, blue jays, rays, patriots, cowboys,
    packers, steelers, broncos, chiefs, raiders, 49ers, niners, seahawks, buccaneers, bucs, vikings, jaguars,
    ravens, bengals, browns, texans, titans, colts, saints, falcons, panthers, eagles, commanders, bears,
    lions, rams, chargers, dolphins, jets, bills, lakers, celtics, warriors, bulls, knicks, mavericks, mavs,
    clippers, nuggets, cavaliers, cavs, pistons, pacers, raptors, pelicans, grizzlies, timberwolves,
    trail blazers, blazers, heat, magic, hawks, hornets, thunder, spurs, suns, kings, jazz, rockets, nets,
    sixers, 76ers, wizards, bucks, bruins, penguins, capitals, blackhawks, red wings, flyers, islanders,
    avalanche, oilers, canucks, maple leafs, canadiens, sabres, predators, kraken, golden knights, lightning,
    hurricanes, devils, stars, wild, blues, ducks, sharks, flames, senators, crimson tide, wolverines,
    buckeyes, longhorns, long horns, sooners, fighting irish, blue devils, tar heels, gators, bulldogs,
    seminoles, hurricanes, tigers, nittany lions, badgers, hawkeyes, huskers, cornhuskers, razorbacks,
    volunteers, gamecocks, aggies, cyclones, jayhawks, wildcats, spartans, hoosiers, the dogs, dawgs"""
)
EMPLOYERS = _csv(
    """walmart, wal-mart, target, costco, kroger, publix, safeway, walgreens, cvs, rite aid, amazon, google,
    microsoft, apple, facebook, meta, tesla, ups, fedex, usps, post office, home depot, lowe's, lowes, ikea,
    starbucks, mcdonald's, mcdonalds, burger king, wendy's, taco bell, chipotle, dunkin, subway, enterprise,
    enterprise rent-a-car, hertz, avis, boeing, lockheed, general motors, gm, ford, toyota, honda, nissan,
    chrysler, bosch, siemens, ge, general electric, ibm, intel, dell, hp, oracle, at&t, verizon, t-mobile,
    comcast, disney, nike, adidas, rei, best buy, kohl's, macy's, nordstrom, jcpenney, sears, kmart, dollar
    general, family dollar, aldi, trader joe's, whole foods, wegmans, heb, h-e-b, meijer, hy-vee, sam's club,
    weight watchers, planet fitness, la fitness, ymca, the y, uber, lyft, doordash, grubhub, instacart,
    infosys, tata, wipro, reliance, apollo, fortis, manipal, max healthcare, narayana, medanta, aiims,
    johnson & johnson, pfizer, merck, abbott, cardinal health, mckesson, landscapes warehouse"""
)
OCCUPATIONS = _csv(
    """teacher, professor, lecturer, lawyer, attorney, judge, engineer, accountant, architect, banker, cashier,
    chef, waiter, waitress, bartender, barista, mechanic, electrician, plumber, carpenter, welder, farmer,
    rancher, trucker, truck driver, bus driver, pilot, flight attendant, police officer, cop, firefighter,
    fireman, paramedic, emt, soldier, veteran, secretary, receptionist, clerk, salesman, saleswoman,
    salesperson, realtor, contractor, construction worker, landscaper, janitor, custodian, housekeeper,
    maid, nanny, babysitter, programmer, software developer, software engineer, developer, consultant,
    analyst, scientist, researcher, librarian, coach, athlete, journalist, reporter, writer, author, editor,
    photographer, artist, musician, singer, actor, actress, designer, hairdresser, hairstylist, barber,
    stylist, cosmetologist, social worker, caseworker, mailman, mail carrier, postal worker, courier,
    foreman, supervisor, executive, ceo, manager, entrepreneur, business owner, pastor, priest, minister,
    rabbi, chaplain, missionary, personal trainer, lifeguard, security guard, bouncer, taxi driver,
    uber driver, delivery driver, bookkeeper, insurance agent, broker, auditor, paralegal, administrator,
    machinist, assembler, home health aide, cna, nursing assistant, line cook, dishwasher, server,
    dental hygienist, hygienist, pharmacist, nurse, rn, lpn, physician, surgeon, dentist, veterinarian,
    vet tech, firefighter, sheriff, deputy, marine, sailor, airman, coal miner, miner, logger, fisherman,
    roofer, painter, mason, bricklayer, ironworker, lineman, technician, dispatcher, cook, baker, butcher,
    tailor, seamstress, cleaner, gardener, groundskeeper, florist, jeweler, pharmacy tech, phlebotomist,
    radiologic technologist, x-ray tech, medical assistant, dental assistant, teacher's aide, substitute
    teacher, principal, dean, student, grad student, college student, retiree"""
)
OCCUPATION_PLACES = _csv(
    """auto repair shop, auto shop, body shop, car dealership, bakery, donut shop, doughnut shop, restaurant,
    diner, bar, coffee shop, cafe, grocery store, convenience store, gas station, warehouse, factory, plant,
    the plant, mill, mine, farm, ranch, construction site, office, law firm, bank, hospital, nursing home,
    school, daycare, salon, barbershop, gym, hotel, casino, call center, shipyard, refinery, landscaping
    business, landscaping company, sports store, store, shop, business, company, firm, practice, clinic"""
)
HOLIDAYS = _csv(
    """christmas, christmas eve, thanksgiving, easter, halloween, new year, new year's, new years, new year's eve,
    fourth of july, 4th of july, independence day, memorial day, labor day, mother's day, father's day,
    valentine's day, valentine's, valentines, super bowl, hanukkah, kwanzaa, passover, ramadan, eid, diwali,
    holi, rosh hashanah, yom kippur, st. patrick's day, saint patrick's day, mardi gras, cinco de mayo"""
)
# Hobby nouns removed by the aggressive preset (polysemous words such as `running`, `lifting`, `cooking`,
# `painting`, `track` are excluded: `running nose`, `lifting the arm`).
ACTIVITY_WORDS = _csv(
    """jogging, hiking, biking, cycling, swimming, golf, golfing,
    golfer, tennis, pickleball, basketball, football, soccer, baseball, softball, volleyball, hockey, lacrosse,
    rugby, bowling, skiing, snowboarding, skating, surfing, kayaking, canoeing, rowing, fishing, hunting,
    camping, rock climbing, boxing, wrestling, karate, judo, yoga, pilates, barre, barre classes, crossfit,
    weightlifting, ballet, gymnastics, gardening, pottery, ceramics, knitting, sewing, woodworking, photography,
    birdwatching, horseback riding, marathon, marathons, triathlon, water polo, cheerleading, cross country,
    sailing, boating, archery, skateboarding, motocross, landscaping, yard work, lawn work, mowing"""
)
BRANDS = _csv(
    """nike, asics, adidas, reebok, new balance, brooks running, hoka, hoka's, saucony, skechers, under armour,
    lululemon, peloton, fitbit, apple watch, iphone, ipad, android, samsung, netflix, facebook, instagram, tiktok,
    youtube, zoom, tesla, harley, harley-davidson, honey nut cheerios, cheerios, oreo, coke, pepsi, mountain dew,
    gatorade, red bull, big mac, whopper, frappuccino, venti, doritos, cheetos, pringles, lay's, freedoes"""
)
DRUG_EXTRA = _csv(
    """tylenol, advil, motrin, aleve, ibuprofen, acetaminophen, naproxen, aspirin, lisinopril, metformin,
    norvasc, amlodipine, protonix, pantoprazole, lasix, furosemide, toprol, metoprolol, crestor, lipitor,
    atorvastatin, rosuvastatin, flovent, albuterol, ultram, tramadol, brilinta, prozac, zoloft, lexapro,
    wellbutrin, xanax, ativan, synthroid, levothyroxine, jardiance, farxiga, ozempic, trulicity, januvia,
    eliquis, xarelto, coumadin, warfarin, plavix, keppra, lyrica, neurontin, gabapentin, flexeril,
    robitussin, mucinex, claritin, zyrtec, allegra, flonase, nasonex, singulair, montelukast, prednisone,
    medrol, augmentin, amoxicillin, keflex, cipro, bactrim, doxycycline, zithromax, z-pack, valtrex,
    imitrex, humira, enbrel, xeljanz, otezla, entresto, aldactone, spironolactone, coreg, carvedilol,
    imdur, nitro, nitroglycerin, wixela, advair, symbicort, breo, spiriva, ventolin, proair, panoxyl, vick,
    vicks, vick's, zarbee, zarbee's, aveeno, aveno, selsun, nizoral, cetaphil, neutrogena, camila, robitussin,
    buspar, buspirone, wellbutrin, cymbalta, effexor, celexa, paxil, trazodone, ambien, adderall, ritalin"""
)

# Local syntax showing the ordinary-word sense of a token that is also a name (`a walker`, `white blood cell`).
WORD_SENSE_PREV = frozenset(
    """a an the this that these those my your his her our their its no any some every each on in at of to for with
    from by about into onto up down i we you they he she it can could would will should might must don't didn't
    doesn't won't is was are were be been being very too more most less birth question check pretty really
    bit little light dark bright pale and or but not just also than as like""".split()
)
WORD_SENSE_NEXT = frozenset(
    """it the that this down them off up out of blood coat cell cells count phlegm tea card line spots stuff color
    colored colour matter house sugar bread rice wine fish meat noise flag light board paper on in my your his her
    their our a an some""".split()
)

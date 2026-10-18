"""Gazetteers and name-extraction helpers shared by the EDA scripts.

Lists are deliberately small and hand-written (stdlib only, no downloads).
They are *analysis* tools to measure hit rates, not the final detector.
Multi-word entries are comma-separated in the source strings.
"""
from __future__ import annotations

import re

from common import *


def _csv(s: str) -> set[str]:
    return {w.strip().lower() for w in s.replace("\n", ",").split(",") if w.strip()}


FIRST_NAMES = set("""
james john robert michael william david richard joseph thomas charles christopher daniel matthew anthony mark donald
steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey ryan jacob gary nicholas eric
jonathan stephen larry justin scott brandon benjamin samuel gregory alexander frank patrick raymond jack dennis jerry
tyler aaron jose adam nathan henry douglas zachary peter kyle noah ethan jeremy walter christian keith roger terry austin
sean gerald carl harold dylan arthur lawrence jordan jesse bryan billy bruce gabriel joe logan alan juan albert willie
elijah wayne randy vincent mason roy ralph eugene russell bobby bradley philip louis harry fred howard carlos
mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy betty sandra margaret ashley kimberly
emily donna michelle carol amanda melissa deborah stephanie dorothy rebecca sharon laura cynthia amy kathleen angela
shirley brenda emma anna pamela nicole samantha katherine christine helen debra rachel carolyn janet maria catherine
heather diane olivia julie joyce victoria ruth virginia lauren kelly christina joan evelyn judith andrea hannah megan
cheryl jacqueline martha madison teresa gloria sara janice ann kathryn abigail sophia frances jean alice judy isabella
julia grace amber denise danielle marilyn beverly charlotte natalie theresa diana brittany doris kayla alexis lori
tiffany kathy louise ella wanda jane sofia audrey monica mildred lucy mia vera betsy irene chloe dawn pat sue bill ray
hope may june april rose lily daisy violet ivy holly summer autumn ginger candy sunny joy faith dakota skye
mike jim tom bob rob dan dave chris matt tony steve ken rick rich jeff tim ron ed jon greg alex sam ben nick josh
andy jake ted ned annie beth liz kate katie cathy pam deb barb sandy cindy mandy jen jenny jess sally
maggie peggy molly polly connie bonnie ronnie johnny jimmy tommy danny eddie freddie
isabel ava eva zoe nora leah mila aria luna ellie stella hazel aurora savannah brooklyn bella
martin jesus angel julio hector cesar oscar ivan rafael diego mateo luis miguel javier ricardo eduardo ramon
elaine marie lynn renee marsha rhonda sheila tracy stacy tammy vicki dana robin kim lynda tonya leslie jamie
""".split())

SURNAMES = set("""
smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez wilson anderson
thomas taylor moore jackson martin lee perez thompson white harris sanchez clark ramirez lewis robinson walker young
allen king wright scott torres nguyen hill flores green adams nelson baker hall rivera campbell mitchell carter roberts
gomez phillips evans turner diaz parker cruz edwards collins reyes stewart morris morales murphy cook rogers gutierrez
ortiz morgan cooper peterson bailey reed kelly howard ramos kim cox ward richardson watson brooks chavez wood james
bennett gray mendoza ruiz hughes price alvarez castillo sanders patel myers long ross foster jimenez powell jenkins
perry russell sullivan bell coleman butler henderson barnes gonzales fisher vasquez simmons romero jordan patterson
alexander hamilton graham reynolds griffin wallace moreno west cole hayes bryant herrera gibson ellis tran medina
aguilar stevens murray ford castro marshall owens harrison fernandez mcdonald woods washington kennedy wells vargas
henry chen freeman webb tucker guzman burns crawford olson simpson porter hunter gordon mendez silva shaw snyder
mason dixon munoz hunt hicks holmes palmer wagner black robertson boyd rose stone salazar fox warren mills meyer
rice schmidt garza daniels ferguson nichols stephens soto weaver ryan gardner payne grant dunn kelley spencer hawkins
arnold pierce vazquez hansen peters santos hart bradley knight elliott cunningham duncan armstrong hudson carroll lane
riley andrews alvarado ray delgado berry perkins hoffman johnston matthews pena richards contreras willis carpenter
lawrence sandoval guerrero george chapman rios estrada ortega watkins greene nunez wheeler valdez harper burke larson
santiago maldonado morrison franklin carlson austin dominguez carr lawson jacobs obrien lynch singh vega bishop
montgomery oliver jensen harvey williamson gilbert dean sims espinoza howell li wong reid hanson le mccoy garrett
burton fuller wang weber welch rojas lucas marquez fields park yang little banks padilla day walsh bowman schultz
luna fowler mejia davidson acosta brewer may holland juarez newman pearson curtis cortez douglas schneider joseph
barrett navarro figueroa keller avila wade molina stanley hopkins campos barnett bates chambers caldwell beck lambert
miranda byrd craig ayala lowe frazier powers neal leonard gregory carrillo sutton fleming rhodes shelton schwartz
norris jennings watts duran weiss gross
""".split())

TITLES_DLG = r"(?:mr|mrs|ms|miss|mister|missus|dr|doctor)"

RELATIVES = _csv("""wife, husband, spouse, partner, boyfriend, girlfriend, fiance, fiancee, mom, mother, mommy, mama, dad,
father, daddy, papa, parents, son, daughter, kid, kids, child, children, baby, brother, brothers, sister, sisters,
sibling, siblings, grandmother, grandma, grandfather, grandpa, granddaughter, grandson, grandkids, grandchildren,
aunt, uncle, cousin, nephew, niece, in-laws, mother-in-law, father-in-law, son-in-law, daughter-in-law, stepmother,
stepfather, stepson, stepdaughter, ex-wife, ex-husband, roommate, neighbor, neighbors, friend, friends, buddy,
coworker, coworkers, colleague, boss, caregiver, twin, twins, godmother, godfather, grandparents""")

OCCUPATIONS = _csv("""teacher, professor, lawyer, attorney, judge, engineer, accountant,
architect, banker, cashier, chef, waiter, waitress, bartender, barista, mechanic, electrician, plumber, carpenter,
welder, farmer, rancher, trucker, truck driver, bus driver, pilot, flight attendant, police officer, cop, firefighter,
paramedic, emt, soldier, veteran, secretary, receptionist, clerk, salesman, salesperson, realtor, contractor,
construction worker, construction, landscaper, landscaping, janitor, custodian, housekeeper, maid, nanny, babysitter,
student, retired, retiree, unemployed, programmer, software developer, consultant, analyst, scientist, researcher,
librarian, coach, athlete, journalist, writer, author, editor, photographer, artist, musician, singer, actor, actress,
designer, hairdresser, barber, stylist, cosmetologist, social worker, caseworker, mailman, mail carrier, postal worker, courier, warehouse, factory, the plant, foreman, supervisor, executive, ceo,
entrepreneur, business owner, self-employed, pastor, priest, minister, rabbi, chaplain, missionary, personal trainer, lifeguard,
security guard, bouncer, taxi driver, uber driver, delivery driver, bookkeeper, insurance agent, broker, auditor,
paralegal, administrator, office job, desk job, machinist, assembler, home health aide, cna, nursing assistant, caregiver,
stay at home mom, stay-at-home, homemaker, housewife, works from home, work from home, remote work, night shift,
day shift, shift work, overnight shift, on my feet all day, manual labor, heavy lifting at work, retail, server,
dishwasher, line cook, fast food, grocery store, auto repair shop, auto shop, body shop, car dealership, i own a,
she owns, he owns, owns a, military, army, navy, marines, air force, coast guard, national guard, enlist, enlisted,
deployed, deployment, boot camp, high school, middle school, elementary school, college, university, grad school,
law school, med school, nursing school, kindergarten, freshman, sophomore, 7th grade, 8th grade, 9th grade,
10th grade, 11th grade, 12th grade, first grade, second grade, third grade, fourth grade, fifth grade, sixth grade,
seventh grade, eighth grade, ninth grade, tenth grade, eleventh grade, twelfth grade""")
OCC_PHRASES = [r"\bwork(?:s|ed|ing)? (?:at|as|for) (?:a |an |the )?(?P<n>[a-z][a-z\-']+(?: [a-z][a-z\-']+){0,2})\b",
               r"\b(?:i'?m|i am|he'?s|she'?s|he is|she is) (?:a |an )?(?:retired |former )?(?P<n>[a-z]+ ?(?:teacher|nurse|driver|officer|worker|engineer|manager|assistant|technician|therapist|student|farmer|mechanic|lawyer|attorney|accountant|chef|cook|clerk|cashier|professor|coach|contractor|landscaper|electrician|plumber|carpenter|painter|pilot|firefighter|paramedic|secretary|receptionist|realtor|consultant|analyst|programmer|designer|writer|photographer|artist|musician|hairdresser|barber|waitress|waiter|bartender|salesman|pastor|minister|veteran|retired))\b",
               r"\b(?:my|his|her) (?:job|boss|coworkers?|shop|business|company|store|restaurant|farm|classroom|students|clients|customers|employer|career|profession)\b",
               r"\b(?:at|from|with) work\b|\bgo(?:ing|es)? (?:back )?to work\b|\bstress at work\b|\bwork(?:ed|s|ing)? outside\b"]

HOBBIES = _csv("""jogging, hiking, biking, cycling, swimming, golf, golfing, tennis, pickleball, basketball, football,
soccer, baseball, softball, volleyball, hockey, lacrosse, rugby, bowling, skiing, snowboarding, skating, surfing,
kayaking, canoeing, rowing, fishing, hunting, camping, rock climbing, climbing, boxing, wrestling, karate, judo,
taekwondo, yoga, pilates, barre, crossfit, weightlifting, weight lifting, treadmill, peloton, zumba, dancing,
dance class, gardening, pottery, ceramics, painting, drawing, knitting, crocheting, sewing, quilting, woodworking,
photography, birdwatching, bird watching, cooking, baking, piano, guitar, drums, violin, singing, choir, theater, chess, poker, bingo, puzzles, video games, gaming, horseback riding, horses, motorcycle, racing, marathon,
triathlon, half marathon, 5k, 10k, deadlifts, deadlift, squats, bench press, cheerleading, gymnastics, cross country, bouldering, sailing, boating, jet ski, scuba diving, snorkeling, archery, trail running,
walking the trails, the trails, spin class, elliptical, lifting weights, lift weights, play catch, coaching, volunteer, volunteering, church choir,
book club, bridge club, garden, gardens, landscaping, yard work, lawn work, mowing, shoveling, crafts, scrapbooking""")
HOBBY_PHRASES = [r"\b(?:play(?:s|ed|ing)?|enjoy(?:s|ed)?|love(?:s|d)?|like(?:s|d)? to|got into|into|hobby|hobbies|avid) (?P<n>(?:a |an |some |the )?[a-z\-']+(?: [a-z\-']+){0,2})\b"]

EVENTS = _csv("""wedding, funeral, party, parties, graduation, vacation, trip, reunion, birthday, anniversary, concert,
tournament, marathon, race, moved here, moved to, moving to, just moved, divorce, divorced, retirement, retiring,
honeymoon, a cruise, on a cruise, cruise ship, festival, parade, prom, homecoming, baptism, bar mitzvah, christmas, thanksgiving, easter,
halloween, holiday, holidays, new year, new year's, new years, fourth of july, 4th of july, memorial day, labor day,
mother's day, father's day, valentine's, valentines, super bowl, playoffs, world series, olympics, election,
house party, house parties, cookout, barbecue, bbq, camping trip, road trip, family reunion, baby shower, bridal shower,
graduation party, retirement party, funeral service, memorial service, game night, date night""")

SPORTS_TEAMS = _csv("""yankees, red sox, dodgers, cubs, braves, astros, phillies, brewers, orioles, patriots, cowboys,
packers, steelers, broncos, chiefs, raiders, 49ers, seahawks, buccaneers, vikings, jaguars, ravens, bengals, lakers,
celtics, warriors, bulls, knicks, mavericks, clippers, nuggets, cavaliers, pistons, pacers, raptors, pelicans,
grizzlies, timberwolves, trail blazers, bruins, penguins, capitals, blackhawks, red wings, flyers, islanders, avalanche,
oilers, canucks, maple leafs, canadiens, sabres, predators, kraken, golden knights, ohio state, alabama crimson,
crimson tide, wolverines, buckeyes, longhorns, sooners, fighting irish, blue devils, tar heels, the eagles game,
the giants game, the bears game, the lions game""")

RELIGION_ETHNICITY = _csv("""church, christian, catholic, protestant, baptist, methodist, lutheran, presbyterian,
episcopal, pentecostal, evangelical, mormon, jewish, judaism, synagogue, muslim, islam, mosque, hindu, buddhist, sikh,
atheist, pastor, priest, rabbi, imam, bible, quran, koran, prayer, praying, sunday school, congregation, hispanic,
latino, latina, african-american, african american, caucasian, asian, native american, american indian, chinese,
japanese, korean, vietnamese, filipino, mexican, puerto rican, cuban, dominican, italian, irish, polish, german,
russian, greek, arab, middle eastern, persian, pakistani, nigerian, ethiopian, somali, jamaican, haitian, brazilian,
colombian, salvadoran, guatemalan, honduran, peruvian, immigrant, immigrated, refugee, citizenship, green card,
interpreter, translator, speaks spanish, speak spanish, speaks english, english is not, native language, white male,
white female, black male, black female, white man, white woman, black man, black woman, hispanic male, hispanic female,
asian male, asian female, caucasian male, caucasian female""")

INSURERS = _csv("""medicare, medicaid, tricare, kaiser, kaiser permanente, aetna, cigna, humana, anthem, bluecross,
blue cross, blue shield, blue cross blue shield, unitedhealthcare, united healthcare, united health, wellcare, molina,
centene, ambetter, champva, va benefits, the va, medicare advantage, obamacare, marketplace plan, hmo, ppo""")

BRANDS = _csv("""nike, asics, adidas, reebok, new balance, brooks, hoka, saucony, skechers, under armour, lululemon,
peloton, fitbit, apple watch, iphone, ipad, android, samsung, google, amazon, walmart, costco, kroger, publix,
safeway, walgreens, cvs, rite aid, starbucks, mcdonald's, mcdonalds, burger king, wendy's, taco bell, chipotle,
dunkin, netflix, facebook, instagram, tiktok, youtube, zoom, uber, lyft, doordash, grubhub, tesla, toyota, honda,
ford, chevy, chevrolet, jeep, subaru, nissan, hyundai, kia, bmw, mercedes, lexus, dodge, harley, enterprise,
enterprise rent-a-car, hertz, avis, fedex, usps, home depot, lowe's, lowes, ikea, cheerios, honey nut cheerios,
oreo, coke, pepsi, gatorade, red bull, weight watchers, noom, nutrisystem, jenny craig, dax, dragon,
vick's, vicks, zarbee's, head & shoulder's, head and shoulders, selsun blue, nizoral, aveeno, aveno, cetaphil,
neutrogena, panoxyl, tumor guy, rex center, rec center, planet fitness, ymca, la fitness, gold's orangetheory""")

VEHICLES = _csv("""car, truck, pickup, pickup truck, suv, van, minivan, motorcycle, bicycle, scooter, moped, boat,
jet ski, atv, four-wheeler, four wheeler, tractor, semi truck, rv, camper, trailer, golf cart, my bike, his bike,
her bike, riding my bike, ride my bike, bike ride, bike riding, mountain bike, road bike""")
PETS = _csv("""my dog, my cat, our dog, our cat, his dog, her dog, his cat, her cat, the dog, a dog, a cat,
my puppy, my kitten, my horse, my horses, my pet, my pets, our pets, my bird, my parrot, my rabbit, my bunny,
my turtle, my chickens, our chickens, my cows, my goats, dog bit, dog bite, cat scratch, cat bite, dog walking,
walk the dog, walking the dog, walks the dog, walk my dog, walking my dog, the puppy, new puppy, pet dog, pet cat""")

PLACES = _csv("""alabama, alaska, arizona, arkansas, california, colorado, connecticut, delaware, florida, georgia,
hawaii, idaho, illinois, indiana, iowa, kansas, kentucky, louisiana, maine, maryland, massachusetts, michigan,
minnesota, mississippi, missouri, montana, nebraska, nevada, new hampshire, new jersey, new mexico, new york,
north carolina, south carolina, north dakota, south dakota, ohio, oklahoma, oregon, pennsylvania, rhode island,
tennessee, texas, utah, vermont, virginia, west virginia, wisconsin, wyoming, boston, chicago, houston, phoenix,
philadelphia, san antonio, san diego, dallas, austin, jacksonville, columbus, charlotte, indianapolis, seattle,
denver, nashville, baltimore, louisville, portland, memphis, milwaukee, albuquerque, tucson, fresno, sacramento,
atlanta, omaha, raleigh, miami, cleveland, tulsa, oakland, minneapolis, wichita, tampa, orlando, pittsburgh,
cincinnati, detroit, las vegas, los angeles, san francisco, san jose, brooklyn, manhattan, the bronx, queens,
honolulu, anchorage, canada, mexico, europe, asia, africa, australia, england, london, paris, france, italy, spain,
germany, ireland, scotland, japan, china, india, korea, vietnam, philippines, thailand, brazil, cuba, jamaica,
bahamas, caribbean, the beach, the lake, the mountains, the coast, the woods, the cabin, the shore, the desert,
overseas, abroad, out of state, out of town, out of the country, down south, up north, back east, out west, midwest,
east coast, west coast, new england, the bay area, downtown, the suburbs, the city, broadway, main street,
culdesac, cul-de-sac, the driveway, the trails, the park, the rec center, the rex center""")
PLACE_PHRASES = [r"\b(?:moved|moving|relocated|trip|vacation|flew|drove|travel(?:ed|ing|led|ling)?|lives?|living|visit(?:ed|ing)?) (?:to|from|in|down to|up to|out to|over to|here from|back from) (?P<n>[a-z][a-z\-']+(?: [a-z][a-z\-']+)?)\b(?= ,| \.| and| for| with| last| this| next| where| which| a | the | in | on |$)",
                 r"\b(?P<n>[a-z][a-z\-']+(?: [a-z][a-z\-']+)?) (?:hospital|medical center|university|church|county|neighborhood|village|island|beach|lake|mountain|river|street|avenue|road|boulevard)\b",
                 r"\b(?:in|from|to) (?P<n>[a-z]+ , (?:georgia|texas|florida|california|ohio|virginia|carolina|york|jersey|michigan|colorado|arizona|nevada|oregon|washington|tennessee|kentucky|indiana|illinois|wisconsin|minnesota|missouri|alabama|louisiana|oklahoma|kansas|iowa|utah|maine|vermont|maryland|delaware|connecticut|massachusetts|pennsylvania|idaho|montana|wyoming|dakota|nebraska|arkansas|mississippi|hawaii|alaska|mexico|hampshire))\b"]
PLACE_HEAD_STOP = {"the", "a", "an", "to", "our", "your", "my", "this", "that", "local", "same", "nearest", "outside", "another",
                   "other", "urgent", "emergency", "primary", "main", "at", "in", "of", "from", "children's", "general",
                   "community", "state", "county", "university", "veterans", "va", "his", "her", "their", "any", "some",
                   "which", "what", "nearby", "closest", "different", "new", "old", "big", "small", "little", "major",
                   "regional", "rural", "city", "private", "public", "teaching", "academic", "outpatient", "inpatient",
                   "psychiatric", "rehab", "rehabilitation", "cancer", "heart", "cardiac", "surgical", "surgery",
                   "medical", "clinic", "hospital", "care", "health", "presents", "present", "presented", "presenting",
                   "go", "going", "went", "come", "came", "back", "down", "up", "over", "out", "here", "there", "into",
                   "visit", "visited", "referred", "admitted", "seen", "see", "saw", "called", "call", "sent", "took",
                   "taken", "transferred", "discharged", "by", "with", "for", "on", "and", "or", "but", "nursing",
                   "home", "senior", "assisted", "living", "skilled", "the hospital", "the clinic", "the office",
                   "office", "practice", "we", "i", "you", "he", "she", "they", "it", "is", "was", "be", "been",
                   "dialysis", "eye", "dental", "pain", "sleep", "wound", "women's", "men's", "family", "walk-in",
                   "walk", "doctor's", "doctors", "doctor", "physician's", "er", "ed", "icu", "trauma", "burn",
                   "ortho", "orthopedic", "orthopedics", "neurology", "cardiology", "oncology", "pediatric", "pediatrics",
                   "mental", "behavioral", "physical", "therapy", "pt", "ot", "ent", "gi", "fertility", "imaging",
                   "radiology", "lab", "labs", "blood", "donation", "plasma", "cvs", "walgreens", "mail", "order",
                   "online", "local pharmacy", "same pharmacy", "your pharmacy", "my pharmacy", "the pharmacy",
                   "kaiser permanente", "a", "near", "nearest", "closer", "close", "far", "left", "right", "across",
                   "whole", "entire", "half", "third", "second", "first", "next", "last", "tech", "state school",
                   "nail", "hair", "coffee", "grocery", "stairs", "stair", "bone", "joint", "spine", "spinal", "chest",
                   "knee", "hip", "shoulder", "neck", "back", "hand", "foot", "ankle", "wrist", "elbow", "head", "heel",
                   "tennis", "golf", "ski", "beach", "lake", "river", "pool", "yoga", "dance", "art", "music", "cooking",
                   "cooking class", "class", "classes", "trade", "night", "day", "summer", "graduate", "law", "med",
                   "business", "nursing", "art school", "the", "that", "this"}

MEDICAL_EPONYMS = _csv("""alzheimer, alzheimer's, parkinson, parkinson's, crohn, crohn's, hodgkin, hodgkin's, graves,
graves', paget, paget's, cushing, cushing's, addison, addison's, lyme, bell, bell's, wilson, wilson's, huntington,
huntington's, lou gehrig, gehrig's, hashimoto, hashimoto's, sjogren, sjogren's, raynaud, raynaud's, barrett,
barrett's, hirschsprung, guillain, barre, marfan, ehlers, danlos, down's, downs, turner, turner's, klinefelter,
kawasaki, still's, achilles, bartholin, bartholin's, baker, baker's, dupuytren, dupuytren's, osgood, schlatter,
de quervain, quervain, quervain's, tourette, tourette's, asperger, asperger's, wegener, wegener's, reiter, reiter's,
whipple, whipple's, meniere, meniere's, colles, colles', lisfranc, jones, morton, morton's, bennett, bennett's, hill,
sachs, kaposi, kaposi's, burkitt, burkitt's, wilms, ewing, ewing's, mallory, weiss, zollinger, ellison, cheyne,
stokes, babinski, homan, homan's, murphy, murphy's, rovsing, mcburney, mcburney's, hawkins, kennedy, neer, lachman,
mcmurray, phalen, tinel, tinel's, finkelstein, finkelstein's, spurling, spurling's, ober, thomas, thompson, apley,
allen, allen's, romberg, glasgow, apgar, braden, morse, epley, dix, hallpike, hawkins-kennedy, foley, kegel, kegels,
heimlich, pap, pap smear, kirschner, salter, harris, gleason, bishop, bishop's, ortolani, barlow, charcot, cheyne-stokes,
holter, doppler, lasik, fowler, trendelenburg, sims, lithotomy, semi-fowler's, fowler's""")

# ---------------------------------------------------------------- name extraction patterns

NOTE_FULLNAME_RE = re.compile(
    r"(?:\b(?P<title>Mr|Mrs|Ms|Miss|Mister)\.? )?(?P<first>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)(?: (?P<last>[A-Z][a-z]+(?:-[A-Z][a-z]+)?))? is an? (?:very )?(?:pleasant |lovely |delightful )?(?:\d+|[a-z]+)[- ](?:year|month|week|day)")
NOTE_TITLE_RE = re.compile(r"\b(?P<title>Mr|Mrs|Ms|Miss|Mister)\.? (?P<last>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)")
NOTE_DR_RE = re.compile(r"\b(?:Dr\.?|Doctor) (?P<last>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)")

DLG_GREET_RE = re.compile(r"\b(?:hi|hello|hey|good morning|good afternoon|good evening|morning|welcome)\b\s?,?\s(?:" + TITLES_DLG + r"\s?\.?\s)?(?P<name>[a-z]+(?:-[a-z]+)?)\b")
DLG_INTRO_RE = re.compile(r"\b(?P<first>[a-z]+)(?: (?P<last>[a-z]+))? is an? (?:very )?(?:pleasant )?(?:\d+|[a-z]+(?: [a-z]+)?) ?-? ?(?:year|month|week|day)s? ?-? ?old")
DLG_HEADER_RE = re.compile(r"\b(?:next patient is|patient's name is|patient is|patient name is|i am seeing|i'm seeing|today i am seeing|here with|this is|seeing) (?P<first>[a-z]+) (?P<last>[a-z]+)(?= ?[,.]| who| she| he| mrn| medical| date| a | is | today)")
DLG_HEADER2_RE = re.compile(r"^(?:\[doctor\] )?(?:okay \. |all right \. |um , |uh , )?(?P<first>[a-z]+) (?P<last>[a-z]+) ?[,.] (?:date of birth|dob|medical record|mrn)", re.M)
DLG_TITLE_RE = re.compile(r"\b(?P<title>" + TITLES_DLG + r")\s?\.?\s(?P<name>[a-z]+(?:-[a-z]+)?)\b")
DLG_VOCATIVE_RE = re.compile(r"(?:^|[,.?!] )(?P<name>[a-z]+) ?(?:[,.?!]|$)", re.M)
DLG_SO_NAME_RE = re.compile(r"\b(?:so|okay|alright|all right|well|thanks|thank you|bye|right|now|and|yeah|great|good|see you)\s?,?\s(?P<name>[a-z]+)\s?[,.?!]")

GREET_STOP = {"there", "everyone", "everybody", "guys", "doc", "doctor", "document", "again", "so", "um", "uh", "how", "what",
              "i", "i'm", "it's", "its", "this", "that", "you", "yeah", "yes", "no", "okay", "ok", "well", "good", "nice",
              "and", "oh", "all", "back", "to", "the", "a", "my", "we", "mom", "dad", "mama", "buddy", "sweetie", "honey",
              "dear", "sir", "ma'am", "miss", "folks", "little", "big", "dragon", "dax", "thank", "thanks", "sorry", "come",
              "let's", "let", "just", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "can", "could",
              "would", "should", "may", "might", "will", "in", "on", "at", "for", "of", "with", "up", "down", "out", "there",
              "everything", "nothing", "something", "it", "he", "she", "they", "them", "me", "us", "him", "her", "his",
              "hi", "hello", "hey", "morning", "afternoon", "glad", "happy", "welcome", "if", "but", "or", "not", "now",
              "then", "when", "where", "why", "who", "which", "one", "two", "ten", "first", "last", "next", "today",
              "tomorrow", "yesterday", "here", "home", "right", "left", "over", "about", "like", "look", "looks",
              "listen", "take", "tell", "go", "get", "got", "give", "see", "say", "said", "know", "think", "feel",
              "um", "uh", "mm", "mm-hmm", "uh-huh", "ah", "gosh", "man", "kid", "kiddo", "sweetheart", "young", "lady",
              "young lady", "young man", "pal", "friend", "y'all", "ya", "there's", "that's", "what's", "how's",
              "let", "lemme", "gon", "wan", "na", "excuse", "pardon", "sure", "absolutely", "definitely", "alright",
              "sir", "miss", "ms", "mr", "mrs", "both", "everybody", "whoever", "anyone"}


def corpus_stopwords(recs, top=600):
    """Most frequent lowercase dialogue tokens, minus known first names/surnames."""
    c = Counter()
    for r in recs:
        c.update(re.findall(r"[a-z]+(?:'[a-z]+)?", r["dialogue"]))
    return {w for w, _ in c.most_common(top)} - FIRST_NAMES - SURNAMES

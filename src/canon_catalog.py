"""
İngiliz Dili ve Edebiyatı Kanon Kataloğu (Canon Catalog)
========================================================
Proje: TÜBİTAK - Evrensel İngiliz Dili ve Edebiyatı Stilometri Atlası
(Chaucer'dan Modernizm'e Tüm Dönemler ve Kanonik Yazarlar)

Bu modül, İngiliz ve Amerikan edebiyatındaki 7 ana edebi dönemi, 37 kanonik yazarı,
Project Gutenberg Book ID'lerini, karakteristik eserlerini ve yerleşik edebi pasajlarını içerir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AuthorEntry:
    """Kanonik bir yazarın künye ve stilometrik profil bilgileri."""
    author_id: str
    name: str
    era: str
    gutenberg_id: int
    canonical_work: str
    century_span: str
    stylistic_profile: str
    sample_passages: List[str] = field(default_factory=list)


# 7 Ana Edebi Dönem
LITERARY_ERAS: List[str] = [
    "Middle English",
    "Renaissance & Early Modern",
    "Enlightenment & 18th Century",
    "Romanticism",
    "Victorian Era",
    "American Classics",
    "Modernism",
]

# 37 Kanonik Yazar Kataloğu
CANON_CATALOG: Dict[str, AuthorEntry] = {
    # --------------------------------------------------------------------------
    # 1. MIDDLE ENGLISH (c. 1100 - 1500)
    # --------------------------------------------------------------------------
    "chaucer": AuthorEntry(
        author_id="chaucer",
        name="Geoffrey Chaucer",
        era="Middle English",
        gutenberg_id=221,
        canonical_work="The Canterbury Tales",
        century_span="14th Century (c. 1343-1400)",
        stylistic_profile="Rhyming couplets, Middle English orthography (whan, shoure), French loanwords, vivid character portraiture.",
        sample_passages=[
            "Whan that Aprille with his shoures soote the droghte of March hath perced to the roote, "
            "and bathed every veyne in swich licour of which vertu engendred is the flour; "
            "whan Zephirus eek with his sweete breeth inspired hath in every holt and heeth "
            "the tendre croppes, and the yonge sonne hath in the Ram his halve cours yronne, "
            "and smale foweles maken melodye, that slepen al the nyght with open ye.",
            "A Knight ther was, and that a worthy man, that fro the tyme that he first bigan "
            "to riden out, he loved chivalrie, trouthe and honour, fredom and curteisie. "
            "Ful worthy was he in his lordes werre, and therto hadde he riden, no man ferre, "
            "as wel in cristendom as in hethenesse, and evere honoured for his worthynesse."
        ]
    ),

    # --------------------------------------------------------------------------
    # 2. RENAISSANCE & EARLY MODERN (c. 1500 - 1660)
    # --------------------------------------------------------------------------
    "shakespeare": AuthorEntry(
        author_id="shakespeare",
        name="William Shakespeare",
        era="Renaissance & Early Modern",
        gutenberg_id=100,
        canonical_work="Hamlet / Sonnets",
        century_span="Late 16th / Early 17th Century",
        stylistic_profile="Blank verse, iambic pentameter, rich soliloquies, archaic pronouns (thou, thy, thine, hath, doth).",
        sample_passages=[
            "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
            "and by opposing end them. To die—to sleep, no more; and by a sleep to say we end "
            "the heart-ache and the thousand natural shocks that flesh is heir to: 'tis a consummation "
            "devoutly to be wish'd. To die, to sleep; to sleep, perchance to dream—ay, there's the rub.",
            "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: "
            "Rough winds do shake the darling buds of May, and summer's lease hath all too short a date; "
            "Sometime too hot the eye of heaven shines, and often is his gold complexion dimm'd; "
            "And every fair from fair sometime declines, by chance or nature's changing course untrimm'd."
        ]
    ),
    "marlowe": AuthorEntry(
        author_id="marlowe",
        name="Christopher Marlowe",
        era="Renaissance & Early Modern",
        gutenberg_id=779,
        canonical_work="Doctor Faustus",
        century_span="Late 16th Century",
        stylistic_profile="Mighty line blank verse, passionate hubris, classical mythology, grand cosmic rhetoric.",
        sample_passages=[
            "Was this the face that launch'd a thousand ships, and burnt the topless towers of Ilium? "
            "Sweet Helen, make me immortal with a kiss! Her lips suck forth my soul; see where it flies! "
            "Come, Helen, come, give me my soul again. Here will I dwell, for heaven is in these lips, "
            "and all is dross that is not Helena. I will be Paris, and for love of thee, "
            "instead of Troy, shall Wittenberg be sack'd.",
            "Ah, Faustus, now hast thou but one bare hour to live, and then thou must be damn'd perpetually! "
            "Stand still, you ever-moving spheres of heaven, that time may cease, and midnight never come; "
            "fair Nature's eye, rise, rise again, and make perpetual day; or let this hour be but a year, "
            "a month, a week, a natural day, that Faustus may repent and save his soul!"
        ]
    ),
    "milton": AuthorEntry(
        author_id="milton",
        name="John Milton",
        era="Renaissance & Early Modern",
        gutenberg_id=20,
        canonical_work="Paradise Lost",
        century_span="17th Century",
        stylistic_profile="Epic grand style, Latinate inversions, complex multi-clause periodic sentences, Biblical gravity.",
        sample_passages=[
            "Of Man's first disobedience, and the fruit of that forbidden tree, whose mortal taste "
            "brought death into the World, and all our woe, with loss of Eden, till one greater Man "
            "restore us, and regain the blissful seat, sing Heavenly Muse, that on the secret top "
            "of Oreb, or of Sinai, didst inspire that shepherd, who first taught the chosen seed, "
            "in the beginning how the Heavens and Earth rose out of Chaos.",
            "Is this the region, this the soil, the clime, said then the lost Archangel, this the seat "
            "that we must change for Heaven, this mournful gloom for that celestial light? "
            "Be it so, since he who now is sovran can dispose and bid what shall be right: "
            "farthest from him is best, whom reason hath equal'd, force hath made supreme "
            "above his equals. Farewell happy fields, where joy for ever dwells!"
        ]
    ),
    "donne": AuthorEntry(
        author_id="donne",
        name="John Donne",
        era="Renaissance & Early Modern",
        gutenberg_id=48772,
        canonical_work="Songs and Sonnets / Holy Sonnets",
        century_span="Early 17th Century",
        stylistic_profile="Metaphysical conceits, intellectual paradoxes, passionate dramatic directness, theological urgency.",
        sample_passages=[
            "Batter my heart, three-person'd God, for you as yet but knock, breathe, shine, and seek to mend; "
            "that I may rise and stand, o'erthrow me, and bend your force to break, blow, burn, and make me new. "
            "I, like an usurp'd town to another due, labour to admit you, but oh, to no end; "
            "Reason, your viceroy in me, me should defend, but is captiv'd, and proves weak or untrue.",
            "Death, be not proud, though some have called thee mighty and dreadful, for thou art not so; "
            "for those whom thou think'st thou dost overthrow die not, poor Death, nor yet canst thou kill me. "
            "From rest and sleep, which but thy pictures be, much pleasure; then from thee much more must flow, "
            "and soonest our best men with thee do go, rest of their bones, and soul's delivery."
        ]
    ),
    "bacon": AuthorEntry(
        author_id="bacon",
        name="Francis Bacon",
        era="Renaissance & Early Modern",
        gutenberg_id=2434,
        canonical_work="The Essays",
        century_span="Late 16th / Early 17th Century",
        stylistic_profile="Aphoristic, terse, balanced antithesis, analytical philosophical prose, pragmatic wisdom.",
        sample_passages=[
            "Studies serve for delight, for ornament, and for ability. Their chief use for delight is in privateness "
            "and retiring; for ornament, is in discourse; and for ability, is in the judgment and disposition of business. "
            "For expert men can execute, and perhaps judge of particulars, one by one; but the general counsels, "
            "and the plots and marshalling of affairs, come best from those that are learned.",
            "Crafty men contemn studies, simple men admire them, and wise men use them; for they teach not their own use; "
            "but that is a wisdom without them, and above them, won by observation. Read not to contradict and confute; "
            "nor to believe and take for granted; nor to find talk and discourse; but to weigh and consider."
        ]
    ),
    "spenser": AuthorEntry(
        author_id="spenser",
        name="Edmund Spenser",
        era="Renaissance & Early Modern",
        gutenberg_id=26301,
        canonical_work="The Faerie Queene",
        century_span="Late 16th Century",
        stylistic_profile="Spenserian stanza, deliberate pseudo-archaisms, allegorical romance, ornate musical cadence.",
        sample_passages=[
            "A gentle Knight was pricking on the plaine, ycladd in mightie armes and silver shielde, "
            "wherein old dints of deepe wounds did remaine, the cruell markes of many a bloudy fielde; "
            "yet armes till that time did he never wield: his angry steede did chide his foming bitt, "
            "as much disdayning to the curbe to yield: full jolly knight he seemd, and faire did sitt, "
            "as one for knightly giusts and fierce encounters fitt.",
            "Upon his brest a bloudie Crosse he bore, the deare remembrance of his dying Lord, "
            "for whose sweete sake that glorious badge he wore, and dead as living ever him ador'd: "
            "upon his shielde the like was also scor'd, for soveraine hope, which in his helpe he had: "
            "right faithfull true he was in deede and word, but of his cheere did seeme too solemne sad."
        ]
    ),

    # --------------------------------------------------------------------------
    # 3. ENLIGHTENMENT & 18TH CENTURY (c. 1660 - 1789)
    # --------------------------------------------------------------------------
    "swift": AuthorEntry(
        author_id="swift",
        name="Jonathan Swift",
        era="Enlightenment & 18th Century",
        gutenberg_id=829,
        canonical_work="Gulliver's Travels / A Modest Proposal",
        century_span="Early 18th Century",
        stylistic_profile="Deadpan satirical irony, clear plain prose style, biting socio-political critique, rational detachment.",
        sample_passages=[
            "My father had a small estate in Nottinghamshire; I was the third of five sons. He sent me to Emanuel College "
            "in Cambridge, at fourteen years old, where I resided three years, and applied myself close to my studies; "
            "but the charge of maintaining me, being too great for a narrow fortune, I was bound apprentice to Mr. James Bates, "
            "an eminent surgeon in London, with whom I continued four years.",
            "Whoever could find out a fair, cheap, and easy method of making these children sound, useful members of the "
            "commonwealth, would deserve so well of the public as to have his statue set up for a preserver of the nation. "
            "I shall now therefore humbly propose my own thoughts, which I hope will not be liable to the least objection.",
            "I desire the reader will observe, that I calculate my remedy for this one individual Kingdom of Ireland, "
            "and for no other that ever was, is, or, I think, ever can be upon Earth. Therefore let no man talk to me of other expedients: "
            "of taxing our absentees at five shillings a pound: of using neither cloaths, nor houshold furniture, except what is of our own growth and manufacture.",
            "I lay down on the grass, which was very short and soft, where I slept sounder than ever I remember to have done in my life, "
            "and, as I reckoned, above nine hours; for when I awaked, it was just daylight. I attempted to rise, but was not able to stir: "
            "for as I happened to lie on my back, I found my arms and legs were strongly fastened on each side to the ground; "
            "and my hair, which was long and thick, tied down in the same manner.",
            "Whoever examines with due circumspection into the Annual Records of Time, will find it remarked, that War is the child "
            "of Pride, and Pride the daughter of Riches. Between the Ancients and Moderns, there was long disputed the dominion of Parnassus."
        ]
    ),
    "defoe": AuthorEntry(
        author_id="defoe",
        name="Daniel Defoe",
        era="Enlightenment & 18th Century",
        gutenberg_id=521,
        canonical_work="Robinson Crusoe",
        century_span="Early 18th Century",
        stylistic_profile="Realistic journalistic detail, pragmatic inventory enumeration, first-person confession, puritan providence.",
        sample_passages=[
            "I was born in the year 1632, in the city of York, of a good family, though not of that country, my father being "
            "a foreigner of Bremen, who settled first at Hull: he got a good estate by merchandise, and leaving off his trade, "
            "lived afterwards at York, from whence he had married my mother, whose relations were named Robinson, a very good family.",
            "I now began to consider seriously upon my condition, and circumstance I was reduced to; and I drew up the state "
            "of my affairs in writing, not so much to leave them to any that were to come after me, for I was like to have but "
            "few heirs, as to deliver my thoughts from daily poring on them, and afflicting my mind."
        ]
    ),
    "pope": AuthorEntry(
        author_id="pope",
        name="Alexander Pope",
        era="Enlightenment & 18th Century",
        gutenberg_id=2428,
        canonical_work="An Essay on Man / The Rape of the Lock",
        century_span="18th Century",
        stylistic_profile="Heroic couplets, neoclassic symmetry, razor-sharp epigrammatic balance, mock-heroic wit.",
        sample_passages=[
            "Know then thyself, presume not God to scan; the proper study of mankind is man. "
            "Placed on this isthmus of a middle state, a being darkly wise, and rudely great: "
            "with too much knowledge for the sceptic side, with too much weakness for the Stoic's pride, "
            "he hangs between; in doubt to act, or rest; in doubt to deem himself a God, or beast.",
            "What dire offence from am'rous causes springs, what mighty contests rise from trivial things, "
            "I sing—This verse to Caryl, Muse! is due: this, ev'n Belinda may vouchsafe to view: "
            "slight is the subject, but not so the praise, if she inspire, and he approve the lays. "
            "Say what strange motive, Goddess! could compel a well-bred Lord to assault a gentle Belle?"
        ]
    ),
    "johnson": AuthorEntry(
        author_id="johnson",
        name="Samuel Johnson",
        era="Enlightenment & 18th Century",
        gutenberg_id=652,
        canonical_work="Rasselas / The Rambler",
        century_span="18th Century",
        stylistic_profile="Polysyllabic Latinate vocabulary, balanced parallel clauses, moral weight, authoritative didacticism.",
        sample_passages=[
            "Ye who listen with credulity to the whispers of fancy, and pursue with eagerness the phantoms of hope; "
            "who expect that age will perform the promises of youth, and that the deficiencies of the present day "
            "will be supplied by the morrow; attend to the history of Rasselas, prince of Abyssinia.",
            "Human life is everywhere a state in which much is to be endured, and little to be enjoyed. "
            "Such is the state of man, that none can be said to be happy; for happiness requires freedom from pain, "
            "and an uninterrupted flow of pleasures; but nature has subjected us to pain, and wisdom forbids us to expect "
            "pleasure without alloy."
        ]
    ),
    "addison": AuthorEntry(
        author_id="addison",
        name="Joseph Addison",
        era="Enlightenment & 18th Century",
        gutenberg_id=12030,
        canonical_work="The Spectator",
        century_span="Early 18th Century",
        stylistic_profile="Gentle urbanity, conversational ease, polished periodic cadence, social observer persona.",
        sample_passages=[
            "I have observed, that a reader seldom peruses a book with pleasure, until he knows whether the writer of it "
            "be a black or a fair man, of a mild or choleric disposition, married or a bachelor, with other particulars of "
            "the like nature, that conduce very much to the right understanding of an author.",
            "Thus I live in the world rather as a spectator of mankind than as one of the species; by which means I have "
            "made myself a speculative statesman, soldier, merchant, and artisan, without ever meddling with any practical "
            "part in life."
        ]
    ),

    # --------------------------------------------------------------------------
    # 4. ROMANTICISM (c. 1789 - 1837)
    # --------------------------------------------------------------------------
    "austen": AuthorEntry(
        author_id="austen",
        name="Jane Austen",
        era="Romanticism",
        gutenberg_id=1342,
        canonical_work="Pride and Prejudice",
        century_span="Early 19th Century",
        stylistic_profile="Free indirect discourse, subtle irony, balanced domestic realism, social manners, polished syntax.",
        sample_passages=[
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
            "However little known the feelings or views of such a man may be on his first entering a neighbourhood, "
            "this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property "
            "of some one or other of their daughters.",
            "Elizabeth Bennet had been at first rather surprised to observe how very little Darcy spoke to any of the party; "
            "and when she remembered his cold reserve in Hertfordshire, she could not but wonder at his choosing to walk with "
            "them at all. But she soon saw that he had no intention of talking; his eyes were fixed on her with an expression "
            "which she could not mistake, though she was very far from being willing to understand it.",
            "She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied "
            "herself nervous. The business of her life was to get her daughters married; its solace was visiting and news.",
            "The family of Dashwood had long been settled in Sussex. Their estate was large, and their residence was at Norland Park, "
            "in the centre of their property, where, for many generations, they had lived in so respectable a manner as to engage the general "
            "good opinion of their surrounding acquaintance. The late owner of this estate was a single man, who lived to a very advanced age.",
            "No one who had ever seen Catherine Morland in her infancy would have supposed her born to be an heroine. Her situation in life, "
            "the character of her father and mother, her own person and disposition, were all equally against her. Her father was a clergyman, "
            "without being neglected, or poor, and a very respectable man, though his name was Richard."
        ]
    ),
    "shelley_mary": AuthorEntry(
        author_id="shelley_mary",
        name="Mary Shelley",
        era="Romanticism",
        gutenberg_id=84,
        canonical_work="Frankenstein",
        century_span="Early 19th Century",
        stylistic_profile="Gothic sublime, epistolary frame narrative, impassioned philosophical torment, vivid scientific hubris.",
        sample_passages=[
            "It was on a dreary night of November that I beheld the accomplishment of my toils. With an anxiety that "
            "almost amounted to agony, I collected the instruments of life around me, that I might infuse a spark of being "
            "into the lifeless thing that lay at my feet. It was already one in the morning; the rain pattered dismally against "
            "the panes, and my candle was nearly burnt out.",
            "I am alone and miserable; man will not associate with me; but one as deformed and horrible as myself would not deny "
            "herself to me. My companion must be of the same species and have the same defects. This being you must create."
        ]
    ),
    "blake": AuthorEntry(
        author_id="blake",
        name="William Blake",
        era="Romanticism",
        gutenberg_id=1934,
        canonical_work="Songs of Innocence and of Experience",
        century_span="Late 18th / Early 19th Century",
        stylistic_profile="Mystic visionary symbolism, incantatory meter, direct repetitive rhythm, prophetic moral outrage.",
        sample_passages=[
            "Tyger Tyger, burning bright, in the forests of the night; what immortal hand or eye, "
            "could frame thy fearful symmetry? In what distant deeps or skies burnt the fire of thine eyes? "
            "On what wings dare he aspire? What the hand dare seize the fire?",
            "I wander thro' each charter'd street, near where the charter'd Thames does flow, "
            "and mark in every face I meet marks of weakness, marks of woe. In every cry of every Man, "
            "in every Infant's cry of fear, in every voice, in every ban, the mind-forg'd manacles I hear."
        ]
    ),
    "wordsworth": AuthorEntry(
        author_id="wordsworth",
        name="William Wordsworth",
        era="Romanticism",
        gutenberg_id=12145,
        canonical_work="Lyrical Ballads / The Prelude",
        century_span="Early 19th Century",
        stylistic_profile="Poetic diction of common speech, recollection of emotion in tranquility, pantheistic nature worship.",
        sample_passages=[
            "I wandered lonely as a cloud that floats on high o'er vales and hills, when all at once I saw a crowd, "
            "a host, of golden daffodils; beside the lake, beneath the trees, fluttering and dancing in the breeze. "
            "Continuous as the stars that shine and twinkle on the milky way, they stretched in never-ending line "
            "along the margin of a bay.",
            "For I have learned to look on nature, not as in the hour of thoughtless youth; but hearing oftentimes "
            "the still, sad music of humanity, nor harsh nor grating, though of ample power to chasten and subdue. "
            "And I have felt a presence that disturbs me with the joy of elevated thoughts; a sense sublime of something "
            "far more deeply interfused, whose dwelling is the light of setting suns."
        ]
    ),
    "keats": AuthorEntry(
        author_id="keats",
        name="John Keats",
        era="Romanticism",
        gutenberg_id=23684,
        canonical_work="Odes and Sonnets",
        century_span="Early 19th Century",
        stylistic_profile="Sensuous tactile imagery, synesthesia, intense melancholy, aesthetic devotion (Beauty is truth).",
        sample_passages=[
            "Thou still unravish'd bride of quietness, thou foster-child of silence and slow time, "
            "Sylvan historian, who canst thus express a flowery tale more sweetly than our rhyme: "
            "What leaf-fring'd legend haunts about thy shape of deities or mortals, or of both, "
            "In Tempe or the dales of Arcady? What men or gods are these? What maidens loth?",
            "My heart aches, and a drowsy numbness pains my sense, as though of hemlock I had drunk, "
            "or emptied some dull opiate to the drains one minute past, and Lethe-wards had sunk: "
            "'Tis not through envy of thy happy lot, but being too happy in thine happiness, "
            "that thou, light-winged Dryad of the trees, in some melodious plot of beechen green."
        ]
    ),
    "byron": AuthorEntry(
        author_id="byron",
        name="Lord Byron",
        era="Romanticism",
        gutenberg_id=21700,
        canonical_work="Don Juan / Childe Harold",
        century_span="Early 19th Century",
        stylistic_profile="Byronic brooding hero, ottava rima cynicism, witty worldly digressions, energetic romantic rebellion.",
        sample_passages=[
            "She walks in beauty, like the night of cloudless climes and starry skies; "
            "and all that's best of dark and bright meet in her aspect and her eyes; "
            "thus mellowed to that tender light which heaven to gaudy day denies. "
            "One shade the more, one ray the less, had half impaired the nameless grace.",
            "I want a hero: an uncommon want, when every year and month sends forth a new one, "
            "till, after cloying the gazettes with cant, the age discovers he is not the true one; "
            "of such as these I should not care to vaunt, I'll therefore take our ancient friend Don Juan."
        ]
    ),
    "coleridge": AuthorEntry(
        author_id="coleridge",
        name="Samuel Taylor Coleridge",
        era="Romanticism",
        gutenberg_id=151,
        canonical_work="The Rime of the Ancient Mariner",
        century_span="Early 19th Century",
        stylistic_profile="Supernatural ballad meter, vivid hallucinatory imagery, moral redemption through nature love.",
        sample_passages=[
            "It is an ancient Mariner, and he stoppeth one of three. 'By thy long grey beard and glittering eye, "
            "now wherefore stopp'st thou me? The Bridegroom's doors are opened wide, and I am next of kin; "
            "the guests are met, the feast is set: may'st hear the merry din.' He holds him with his skinny hand, "
            "'There was a ship,' quoth he.",
            "Water, water, everywhere, and all the boards did shrink; water, water, everywhere, nor any drop to drink. "
            "The very deep did rot: O Christ! that ever this should be! Yea, slimy things did crawl with legs upon the slimy sea."
        ]
    ),

    # --------------------------------------------------------------------------
    # 5. VICTORIAN ERA (c. 1837 - 1901)
    # --------------------------------------------------------------------------
    "dickens": AuthorEntry(
        author_id="dickens",
        name="Charles Dickens",
        era="Victorian Era",
        gutenberg_id=98,
        canonical_work="A Tale of Two Cities / Great Expectations / Bleak House",
        century_span="Mid 19th Century",
        stylistic_profile="Anaphora, vivid grotesque caricature, rhythmic cumulative clauses, profound social pathos.",
        sample_passages=[
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
            "it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, "
            "it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, "
            "we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like "
            "the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, "
            "in the superlative degree of comparison only.",
            "Fog everywhere. Fog up the river, where it flows among green aits and meadows; fog down the river, where it rolls "
            "defiled among the tiers of shipping, and the waterside pollutions of a great and dirty city. Fog on the Essex marshes, "
            "fog on the Kentish heights. Fog creeping into the cabooses of collier-brigs; fog lying out on the yards, and hovering in "
            "the rigging of great ships; fog drooping on the gunwales of barges and small boats. Fog in the eyes and throats of ancient "
            "Greenwich pensioners, wheezing by the firesides of their wards; fog in the stem and bowl of the afternoon pipe of the "
            "wrathful skipper, down in his close cabin.",
            "My father's family name being Pirrip, and my christian name Philip, my infant tongue could make of both names nothing "
            "longer or more explicit than Pip. So, I called myself Pip, and came to be called Pip. I give Pirrip as my father's "
            "family name, on the authority of his tombstone and my sister—Mrs. Joe Gargery, who married the blacksmith. Ours was "
            "the marsh country, down by the river, within, as the river wound, twenty miles of the sea.",
            "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, "
            "these pages must show. To begin my life with the beginning of my life, I record that I was born (as I have been informed "
            "and believe) on a Friday, at twelve o'clock at night. It was remarked that the clock began to strike, and I began to cry, "
            "simultaneously.",
            "Marley was dead: to begin with. There is no doubt whatever about that. The register of his burial was signed by the clergyman, "
            "the clerk, the undertaker, and the chief mourner. Scrooge signed it: and Scrooge's name was good upon 'Change, for anything he "
            "chose to put his hand to. Old Marley was as dead as a door-nail. Mind! I don't mean to say that I know, of my own knowledge, "
            "what there is particularly dead about a door-nail."
        ]
    ),
    "wilde": AuthorEntry(
        author_id="wilde",
        name="Oscar Wilde",
        era="Victorian Era",
        gutenberg_id=174,
        canonical_work="The Picture of Dorian Gray",
        century_span="Late 19th Century",
        stylistic_profile="Aesthetic decadent luxury, epigrammatic paradoxical wit, polished conversational insolence, sensual descriptions.",
        sample_passages=[
            "The studio was filled with the rich odour of roses, and when the light summer wind stirred amidst the trees of the garden, "
            "there came through the open door the heavy scent of the lilac, or the more delicate perfume of the pink-flowering thorn. "
            "From the corner of the divan of Persian saddle-bags on which he was lying, Lord Henry Wotton could catch the gleam.",
            "The only way to get rid of a temptation is to yield to it. Resist it, and your soul grows sick with longing for the things "
            "it has forbidden to itself, with desire for what its monstrous laws have made monstrous and unlawful. It has been said that "
            "the great events of the world take place in the brain."
        ]
    ),
    "bronte_charlotte": AuthorEntry(
        author_id="bronte_charlotte",
        name="Charlotte Brontë",
        era="Victorian Era",
        gutenberg_id=1260,
        canonical_work="Jane Eyre",
        century_span="Mid 19th Century",
        stylistic_profile="Intense first-person moral agency, proto-feminist defiance, Gothic atmospheric realism, spiritual autonomy.",
        sample_passages=[
            "There was no possibility of taking a walk that day. We had been wandering, indeed, in the leafless shrubbery an hour "
            "in the morning; but since dinner (Mrs. Reed, when there was no company, dined early) the cold winter wind had brought "
            "with it clouds so sombre, and a rain so penetrating, that further outdoor exercise was now out of the question.",
            "Do you think I am an automaton?—a machine without feelings? and can bear to have my morsel of bread snatched from my lips, "
            "and my drop of living water dashed from my cup? Do you think, because I am poor, obscure, plain, and little, I am soulless "
            "and heartless? You think wrong!—I have as much soul as you,—and full as much heart!"
        ]
    ),
    "bronte_emily": AuthorEntry(
        author_id="bronte_emily",
        name="Emily Brontë",
        era="Victorian Era",
        gutenberg_id=768,
        canonical_work="Wuthering Heights",
        century_span="Mid 19th Century",
        stylistic_profile="Elemental passionate savagery, moorland sublime, double frame narrative, supernatural romantic obsession.",
        sample_passages=[
            "1801.—I have just returned from a visit to my landlord—the solitary neighbour that I shall be troubled with. "
            "This is certainly a beautiful country! In all England, I do not believe that I could have fixed on a situation "
            "so completely removed from the stir of society. A perfect misanthropist's heaven: and Mr. Heathcliff and I "
            "were such a suitable pair to divide the desolation between us.",
            "Whatever our souls are made of, his and mine are the same; and Linton's is as different as a moonbeam from lightning, "
            "or frost from fire. My love for Heathcliff resembles the eternal rocks beneath: a source of little visible delight, "
            "but necessary. Nelly, I am Heathcliff! He's always, always in my mind: not as a pleasure, any more than I am always "
            "a pleasure to myself, but as my own being."
        ]
    ),
    "eliot_george": AuthorEntry(
        author_id="eliot_george",
        name="George Eliot",
        era="Victorian Era",
        gutenberg_id=145,
        canonical_work="Middlemarch",
        century_span="Mid-to-Late 19th Century",
        stylistic_profile="Psychological realism, philosophical narrator reflections, sympathetic humanism, intricate web of social provincial life.",
        sample_passages=[
            "Miss Brooke had that kind of beauty which seems to be thrown into relief by poor dress. Her hand and wrist were so finely "
            "formed that she could wear sleeves not less bare of style than those in which the Blessed Virgin appeared to Italian painters; "
            "and her profile as well as her stature and bearing seemed to gain the more dignity from her plain garments.",
            "For the growing good of the world is partly dependent on unhistoric acts; and that things are not so ill with you and me "
            "as they might have been, is half owing to the number who lived faithfully a hidden life, and rest in unvisited tombs."
        ]
    ),
    "hardy": AuthorEntry(
        author_id="hardy",
        name="Thomas Hardy",
        era="Victorian Era",
        gutenberg_id=110,
        canonical_work="Tess of the d'Urbervilles",
        century_span="Late 19th Century",
        stylistic_profile="Deterministic naturalism, rural Wessex dialect, somber architectural imagery, tragic cosmic irony.",
        sample_passages=[
            "On an evening in the latter part of May a middle-aged man was walking homeward from Shaston to the village of Marlott, "
            "in the adjoining Vale of Blakemore, or Blackmoor. The pair of legs that carried him were rickety, and there was a bias "
            "in his gait which inclined him somewhat to the left of a straight line.",
            "The broad experience of the world had taught her that, in the complex tragedy of human life, pity is very close to love, "
            "and sorrow the bitterest solvent of pride. She felt that she had been wronged, but she also felt that she had sinned; "
            "and nature took no heed of her intentions, pursuing its inexorable course with stony indifference."
        ]
    ),
    "doyle": AuthorEntry(
        author_id="doyle",
        name="Arthur Conan Doyle",
        era="Victorian Era",
        gutenberg_id=1661,
        canonical_work="The Adventures of Sherlock Holmes",
        century_span="Late 19th Century",
        stylistic_profile="Watson's companionable narrative voice, crisp deductive analytical dialogue, Victorian London fog atmosphere.",
        sample_passages=[
            "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. In his eyes "
            "she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. "
            "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.",
            "You see, but you do not observe. The distinction is clear. For example, you have frequently seen the steps which lead up "
            "from the hall to this room. How many are there? You do not know. Quite so! And yet you have seen them. I knew that there "
            "were seventeen steps, because I have both seen and observed."
        ]
    ),
    "stoker": AuthorEntry(
        author_id="stoker",
        name="Bram Stoker",
        era="Victorian Era",
        gutenberg_id=345,
        canonical_work="Dracula",
        century_span="Late 19th Century",
        stylistic_profile="Epistolary journals and telegrams, Gothic dread, East-West clash, hypnotic vampiric allure.",
        sample_passages=[
            "3 May. Bistritz.—Left Munich at 8:35 P. M., on 1st May, arriving at Vienna early next morning; should have arrived at 6:46, "
            "but train was an hour late. Buda-Pesth seems a wonderful place, from the glimpse which I got of it from the train and the little "
            "I could walk through the streets. I feared to go very far from the station, as we had arrived late.",
            "Welcome to my house! Enter freely and of your own will! He made no motion of stepping to meet me, but stood like a statue, "
            "as though his gesture of welcome had fixed him into stone. The moment, however, that I had stepped over the threshold, "
            "he moved impulsively forward, and holding out his hand grasped mine with a strength which made me wince."
        ]
    ),

    # --------------------------------------------------------------------------
    # 6. AMERICAN CLASSICS (c. 1830 - 1910)
    # --------------------------------------------------------------------------
    "poe": AuthorEntry(
        author_id="poe",
        name="Edgar Allan Poe",
        era="American Classics",
        gutenberg_id=2148,
        canonical_work="The Raven / Tales of the Grotesque",
        century_span="Mid 19th Century",
        stylistic_profile="Macabre psychological terror, incantatory musical rhythm, unity of effect, morbid obsession.",
        sample_passages=[
            "Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume of forgotten lore— "
            "While I nodded, nearly napping, suddenly there came a tapping, as of some one gently rapping, rapping at my chamber door. "
            "'Tis some visitor,' I muttered, 'tapping at my chamber door—only this and nothing more.'",
            "True!—nervous—very, very dreadfully nervous I had been and am; but why will you say that I am mad? The disease had sharpened "
            "my senses—not destroyed—not dulled them. Above all was the sense of hearing acute. I heard all things in the heaven and in "
            "the earth. I heard many things in hell. How, then, am I mad? Hearken! and observe how healthily—how calmly I can tell you the whole story."
        ]
    ),
    "melville": AuthorEntry(
        author_id="melville",
        name="Herman Melville",
        era="American Classics",
        gutenberg_id=2701,
        canonical_work="Moby Dick",
        century_span="Mid 19th Century",
        stylistic_profile="Biblical cetological encyclopedism, metaphysical Shakespearean monologues, maritime philosophical grandeur.",
        sample_passages=[
            "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing "
            "particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. "
            "It is a way I have of driving off the spleen and regulating the circulation.",
            "He piled upon the whale's white hump the sum of all the general rage and hate felt by his whole race from Adam down; "
            "and then, as if his chest had been a mortar, he burst his hot heart's shell upon it. Towards thee I roll, thou all-destroying "
            "but unconquering whale; to the last I grapple with thee; from hell's heart I stab at thee; for hate's sake I spit my last breath at thee."
        ]
    ),
    "twain": AuthorEntry(
        author_id="twain",
        name="Mark Twain",
        era="American Classics",
        gutenberg_id=76,
        canonical_work="Adventures of Huckleberry Finn",
        century_span="Late 19th Century",
        stylistic_profile="Vernacular spoken dialect, deadpan frontier humor, moral satire against hypocrisy, Mississippi realism.",
        sample_passages=[
            "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter. "
            "That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he "
            "told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly.",
            "It was mighty still cool and pleasant. The raft was going along down the river, and the banks was all quiet and asleep; "
            "and sometimes we'd hear a steer bell away off, and it was so still you could hear it a mile. There was kind of a solemnity "
            "about the big river in the dark."
        ]
    ),
    "whitman": AuthorEntry(
        author_id="whitman",
        name="Walt Whitman",
        era="American Classics",
        gutenberg_id=1322,
        canonical_work="Leaves of Grass",
        century_span="Mid-to-Late 19th Century",
        stylistic_profile="Free verse, sweeping operatic catalogs, democratic egoism, ecstatic physical celebration of body and nation.",
        sample_passages=[
            "I celebrate myself, and sing myself, and what I assume you shall assume, for every atom belonging to me as good belongs to you. "
            "I loafe and invite my soul, I lean and loafe at my ease observing a spear of summer grass. My tongue, every atom of my blood, "
            "form'd from this soil, this air, born here of parents born here from parents the same, and their parents the same.",
            "Do I contradict myself? Very well then I contradict myself, (I am large, I contain multitudes.) I concentrate toward them "
            "that are nigh, the wait on the door-sill. Who has done his day's work? who will soonest be through with his supper? "
            "Who wishes to walk with me? Will you speak before I am gone? will you prove already too late?"
        ]
    ),
    "dickinson": AuthorEntry(
        author_id="dickinson",
        name="Emily Dickinson",
        era="American Classics",
        gutenberg_id=12242,
        canonical_work="Collected Poems",
        century_span="Mid 19th Century",
        stylistic_profile="Characteristic dashes (—), ballad meter variation, slant rhyme, telegraphic compressed epigrammatic mysticism.",
        sample_passages=[
            "Because I could not stop for Death—he kindly stopped for me—the Carriage held but just Ourselves—and Immortality. "
            "We slowly drove—he knew no haste and I had put away my labor and my leisure too, for his Civility— "
            "We passed the School, where Children strove at Recess—in the Ring—We passed the Fields of Gazing Grain—We passed the Setting Sun—",
            "'Hope' is the thing with feathers—that perches in the soul—and sings the tune without the words—and never stops—at all— "
            "and sweetest—in the Gale—is heard—and sore must be the storm—that could abash the little Bird that kept so many warm—"
        ]
    ),
    "hawthorne": AuthorEntry(
        author_id="hawthorne",
        name="Nathaniel Hawthorne",
        era="American Classics",
        gutenberg_id=33,
        canonical_work="The Scarlet Letter",
        century_span="Mid 19th Century",
        stylistic_profile="Puritan guilt allegories, moral ambiguity, dark romantic psychological introspection, ornate formal diction.",
        sample_passages=[
            "The door of the jail being flung open from within, there appeared, in the first place, like a black shadow emerging into "
            "sunshine, the grim and grisly presence of the town-beadle, with a sword by his side, and his staff of office in his hand. "
            "This personage prefigured and represented in his aspect the whole dismal severity of the Puritanic code of law.",
            "On the breast of her gown, in fine red cloth, surrounded with an elaborate embroidery and fantastic flourishes of gold-thread, "
            "appeared the letter A. It was so artistically done, and with so much fertility and gorgeous luxuriance of fancy, "
            "that it had all the effect of a last and fitting decoration to the apparel which she wore."
        ]
    ),

    # --------------------------------------------------------------------------
    # 7. MODERNISM (c. 1900 - 1950)
    # --------------------------------------------------------------------------
    "woolf": AuthorEntry(
        author_id="woolf",
        name="Virginia Woolf",
        era="Modernism",
        gutenberg_id=144,
        canonical_work="Mrs. Dalloway / To the Lighthouse",
        century_span="Early 20th Century",
        stylistic_profile="Stream of consciousness, parenthetical impressions, rhythmic lyrical interiority, temporal fluidity (moments of being).",
        sample_passages=[
            "Mrs. Dalloway said she would buy the flowers herself. For Lucy had her work cut out for her. The doors would be taken off "
            "their hinges; Rumpelmayer's men were coming. And then, thought Clarissa Dalloway, what a morning—fresh as if issued to children on a beach. "
            "What a lark! What a plunge! For so it had always seemed to her, when, with a little squeak of the hinges, which she could hear now, "
            "she had burst open the French windows and plunged at Bourton into the open air.",
            "How fresh, how calm, stiller than this of course, the air was in the early morning; like the flap of a wave; the kiss of a wave; "
            "chill and sharp and yet (for a girl of eighteen as she then was) solemn, feeling as she did, standing there at the open window, "
            "that something awful was about to happen; looking at the flowers, at the trees with the smoke winding off them."
        ]
    ),
    "joyce": AuthorEntry(
        author_id="joyce",
        name="James Joyce",
        era="Modernism",
        gutenberg_id=4300,
        canonical_work="Ulysses / Dubliners",
        century_span="Early 20th Century",
        stylistic_profile="Radical stream of consciousness, linguistic paronomasia, sensory immediacy, cataloging of epiphanies.",
        sample_passages=[
            "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed. "
            "A yellow dressinggown, ungirdled, was sustained gently behind him on the mild morning air. He held the bowl aloft and intoned: "
            "—Introibo ad altare Dei. Halted, he peered down the dark winding stairs and called out coarsely: —Come up, Kinch! Come up, you fearful jesuit!",
            "Mr Leopold Bloom ate with relish the inner organs of beasts and fowls. He liked thick giblet soup, nutty gizzards, "
            "a stuffed roast heart, liverslices fried with crustcrumbs, fried hencods' roes. Most of all he liked grilled mutton kidneys "
            "which gave to his palate a fine tang of faintly scented urine."
        ]
    ),
    "orwell": AuthorEntry(
        author_id="orwell",
        name="George Orwell",
        era="Modernism",
        gutenberg_id=72260,
        canonical_work="1984 / Animal Farm",
        century_span="Mid 20th Century",
        stylistic_profile="Window-pane clarity, political prose as art, anti-totalitarian stark realism, clinical precision.",
        sample_passages=[
            "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast "
            "in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough "
            "to prevent a swirl of gritty dust from entering along with him.",
            "The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, "
            "had been tacked to the wall. It depicted simply an enormous face, more than a metre wide: the face of a man of about forty-five, "
            "with a heavy black moustache and ruggedly handsome features. BIG BROTHER IS WATCHING YOU, the caption said.",
            "All animals are equal, but some animals are more equal than others. The creatures outside looked from pig to man, "
            "and from man to pig, and from pig to man again; but already it was impossible to say which was which. "
            "Political language is designed to make lies sound truthful and murder respectable, and to give an appearance of solidity to pure wind.",
            "Doublethink means the power of holding two contradictory beliefs in one's mind simultaneously, and accepting both of them. "
            "War is peace, freedom is slavery, ignorance is strength. If liberty means anything at all, it means the right to tell people what they do not want to hear."
        ]
    ),
    "conrad": AuthorEntry(
        author_id="conrad",
        name="Joseph Conrad",
        era="Modernism",
        gutenberg_id=219,
        canonical_work="Heart of Darkness",
        century_span="Turn of 20th Century",
        stylistic_profile="Impressionistic prose, delayed decoding, brooding maritime narrative frame, haunting psychological atmosphere.",
        sample_passages=[
            "The Nellie, a cruising yawl, swung to her anchor without a flutter of the sails, and was at rest. The flood had made, "
            "the wind was nearly calm, and being bound down the river, the only thing for it was to come to and wait for the turn of the tide. "
            "The sea-reach of the Thames stretched before us like the beginning of an interminable waterway.",
            "Going up that river was like traveling back to the earliest beginnings of the world, when vegetation rioted on the earth "
            "and the big trees were kings. An empty stream, a great silence, an impenetrable forest. The air was warm, thick, heavy, sluggish. "
            "There was no joy in the brilliance of sunshine. The long stretches of the waterway ran on, deserted, into the gloom of overshadowed distances."
        ]
    ),
    "yeats": AuthorEntry(
        author_id="yeats",
        name="William Butler Yeats",
        era="Modernism",
        gutenberg_id=8905,
        canonical_work="The Tower / Selected Poems",
        century_span="Late 19th / Early 20th Century",
        stylistic_profile="Symbolist mythology, incisive apocalyptic modernism, high formal gravity, Irish nationalist ethos.",
        sample_passages=[
            "Turning and turning in the widening gyre the falcon cannot hear the falconer; things fall apart; the centre cannot hold; "
            "mere anarchy is loosed upon the world, the blood-dimmed tide is loosed, and everywhere the ceremony of innocence is drowned; "
            "the best lack all conviction, while the worst are full of passionate intensity.",
            "That is no country for old men. The young in one another's arms, birds in the trees—those dying generations—at their song, "
            "the salmon-falls, the mackerel-crowded seas, fish, flesh, or fowl, commend all summer long whatever is begotten, born, and dies. "
            "Caught in that sensual music all neglect monuments of unageing intellect."
        ]
    ),
    "eliot_ts": AuthorEntry(
        author_id="eliot_ts",
        name="T. S. Eliot",
        era="Modernism",
        gutenberg_id=1321,
        canonical_work="The Waste Land / Prufrock",
        century_span="Early 20th Century",
        stylistic_profile="Fragmented cultural polyphony, collage of myth, dry intellectual disillusionment, revolutionary free-verse jazz cadence.",
        sample_passages=[
            "April is the cruellest month, breeding lilacs out of the dead land, mixing memory and desire, stirring dull roots with spring rain. "
            "Winter kept us warm, covering earth in forgetful snow, feeding a little life with dried tubers. Summer surprised us, coming over the Starnbergersee "
            "with a shower of rain; we stopped in the colonnade, and went on in sunlight, into the Hofgarten, and drank coffee, and talked for an hour.",
            "Let us go then, you and I, when the evening is spread out against the sky like a patient etherized upon a table; "
            "let us go, through certain half-deserted streets, the muttering retreats of restless nights in one-night cheap hotels "
            "and sawdust restaurants with oyster-shells: streets that follow like a tedious argument of insidious intent."
        ]
    ),
    "fitzgerald": AuthorEntry(
        author_id="fitzgerald",
        name="F. Scott Fitzgerald",
        era="Modernism",
        gutenberg_id=64317,
        canonical_work="The Great Gatsby",
        century_span="1920s Jazz Age",
        stylistic_profile="Lyrical romantic nostalgia, evocative sensory excess, melancholy critique of the American Dream.",
        sample_passages=[
            "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since. "
            "'Whenever you feel like criticizing any one,' he told me, 'just remember that all the people in this world haven't had "
            "the advantages that you've had.' He didn't say any more, but we've always been unusually communicative in a reserved way.",
            "Gatsby believed in the green light, the orgastic future that year by year recedes before us. It eluded us then, "
            "but that's no matter—tomorrow we will run faster, stretch out our arms farther. . . . And one fine morning—— "
            "So we beat on, boats against the current, borne back ceaselessly into the past."
        ]
    ),
    "hemingway": AuthorEntry(
        author_id="hemingway",
        name="Ernest Hemingway",
        era="Modernism",
        gutenberg_id=67138,
        canonical_work="In Our Time / The Sun Also Rises",
        century_span="1920s-1930s",
        stylistic_profile="Iceberg theory, extreme parataxis (repetitive 'and'), terse declarative cadence, deliberate omission of adjectives.",
        sample_passages=[
            "In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains. "
            "In the bed of the river there were pebbles and boulders, dry and white in the sun, and the water was clear and swiftly moving "
            "and blue in the channels. Troops went by the house and down the road and the dust they raised powdered the leaves of the trees.",
            "The hills across the valley of the Ebro were long and white. On this side there was no shade and no trees and the station "
            "was between two lines of rails in the sun. Close against the side of the station there was the warm shadow of the building "
            "and a curtain, made of strings of bamboo beads, hung across the open door into the bar, to keep out flies."
        ]
    ),
}


def get_authors_by_era(era_name: str) -> List[AuthorEntry]:
    """Belirtilen edebi döneme ait yazarları döndürür."""
    return [entry for entry in CANON_CATALOG.values() if entry.era.lower() == era_name.lower()]


def get_all_eras() -> List[str]:
    """Tüm dönem isimlerini kronolojik sırayla döndürür."""
    return list(LITERARY_ERAS)


def get_author_by_id(author_id: str) -> Optional[AuthorEntry]:
    """ID ile yazar nesnesini çeker."""
    return CANON_CATALOG.get(author_id.lower())

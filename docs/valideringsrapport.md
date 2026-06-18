# Valideringsrapport

Rapporten har to deler. Først kommer planen for evalueringen, og deretter resultatene målt mot planen. Planen for søk ble skrevet ferdig før søkene ble kjørt på testsettet, og den endres ikke etterpå. Planen for svar og vurdering skrives før svarene lages for testsettet.

## Plan for søk

### Data

Søkene måles på utviklingssettet og testsettet i `testsett/`. Bare spørsmål som har et svar i korpuset er med, fordi de ubesvarbare spørsmålene ikke har noen riktige paragrafer. Det gir 85 spørsmål i utviklingssettet og 198 i testsettet. I testsettet er 66 direkte spørsmål, 89 på hverdagsspråk og 43 krysshenvisninger. 27 av dem er skrevet på nynorsk.

De riktige paragrafene for hvert spørsmål kalles gullparagrafer. Et spørsmål om krysshenvisning har to. Alt som skal velges, velges på utviklingssettet. Testsettet kjøres én gang med valgene som står her.

### Korpus og tekstbiter

Hvert system kjøres på to korpus med de samme spørsmålene: skattekorpuset med 9 dokumenter og hele korpuset med 4 210 dokumenter. Opphevede og tomme paragrafer er ikke med.

Teksten deles opp på to måter. Strukturbiter følger loven. Hver bit er én paragraf, med lovens navn, kapittel og paragraftittel foran teksten. En paragraf med mer enn 300 ord deles ved grensene mellom leddene. Ledd som ikke hører til noen paragraf, samles per dokument. Vinduer ser bort fra strukturen. Hvert vindu har 200 ord uten overskrift, og et nytt vindu starter hvert 150. ord.

| Tekstbiter | Skattekorpuset | Hele korpuset |
|---|--:|--:|
| Strukturbiter | 2 247 | 127 657 |
| Vinduer | 1 527 | 88 223 |

### Fra tekstbiter til paragrafer

Et søk gir en rangert liste med tekstbiter. Listen gjøres om til en liste med paragrafer uten gjentakelser, og det er denne listen som måles. Tekst utenfor paragrafer telles per ledd.

Et vindu kan dekke flere paragrafer. Da kommer paragrafen med flest ord i vinduet først. Denne regelen ble innført etter den første kjøringen på utviklingssettet. Før kom den første paragrafen i vinduet først. Da telte et vindu som bare begynte med slutten av en paragraf, mest for den paragrafen. Regelen gjelder bare vinduene og endrer ikke teksten i dem.

### Systemer

| System | Hva det gjør |
|---|---|
| BM25 | Ordsøk med små bokstaver, norsk ordstamming (Snowball) og stoppord. |
| BM25 med orddeling | Som BM25, men lange sammensatte ord deles også i delene sine. Hver del må finnes ofte nok i korpuset, og delene kan bindes sammen med s eller e. |
| Vektorsøk | Fire modeller: `intfloat/multilingual-e5-base`, `BAAI/bge-m3`, `NbAiLab/nb-sbert-v2-base` og `NbAiLab/nb-sbert-v2-large`. Vektorene er normalisert og sammenlignes med skalarprodukt. |
| Hybrid | Resiprok rangfusjon (k = 60) av det beste ordsøket og det beste vektorsøket. |
| Omrangering | `BAAI/bge-reranker-v2-m3` ordner de 30 første tekstbitene fra hybriden på nytt. |
| Utvidelse | Etter de fem første paragrafene kommer paragrafene de viser til, i den rekkefølgen de er nevnt. Deretter kommer resten av listen. |

Hybriden ble valgt på utviklingssettet etter nDCG@10. På skattekorpuset består den av BM25 og bge-m3. På hele korpuset består den av BM25 med orddeling og e5-base. Der er e5-base den eneste vektormodellen, så valget gjaldt bare ordsøket. Omrangering brukes bare på strukturbiter, for å spare regnetid. Utvidelse brukes på hybriden og på hybriden med omrangering.

### Hele korpuset

All vektorberegning skjer på prosessoren (CPU) i en bærbar PC. Der lager e5-base omtrent tre vektorer i sekundet, og bge-m3 omtrent én. Med 127 657 strukturbiter ville hver av de største modellene trengt mer enn et døgn. Derfor er bare e5-base kjørt på hele korpuset, og bare på strukturbiter. Av de to minste modellene var e5-base best på utviklingssettet. På vinduene i hele korpuset er bare ordsøk kjørt. Sammenligningen mellom korpusene er derfor renest for ordsøk og for e5-base.

### Mål

For hvert spørsmål måles recall@1, recall@5, recall@10, MRR og nDCG@10. En paragraf er enten en gullparagraf eller ikke, uten grader mellom. Recall@k er andelen av gullparagrafene som er med blant de k første paragrafene. Tallene er gjennomsnitt over spørsmålene, med et 95 prosent konfidensintervall fra bootstrap med 2 000 trekninger og fast frø. Resultatene vises for alle spørsmål, per type spørsmål og per skriftform. I tillegg måles tiden det tar å lage vektorene og størrelsen på indeksen.

### Sammenligninger

Hovedmålet er nDCG@10. Sammenligningene under er bestemt på forhånd, og den nøyaktige listen står under `retrieval.comparisons` i `configs/default.yaml`. For hver sammenligning regnes forskjellen per spørsmål. Intervallet lages ved å trekke de samme spørsmålene for begge systemene. En forskjell regnes som tydelig når hele intervallet ligger på samme side av null.

1. Hybriden mot ordsøket og vektorsøket den er bygd av, på begge korpusene.
2. Omrangering mot hybriden, på begge korpusene.
3. BM25 med orddeling mot vanlig BM25 på skattekorpuset.
4. Norske mot flerspråklige modeller av samme størrelse: nb-sbert-v2-base mot e5-base, og nb-sbert-v2-large mot bge-m3.
5. Utvidelse mot ingen utvidelse. På skattekorpuset gjelder det alle spørsmål og krysshenvisninger alene. På hele korpuset gjelder det krysshenvisninger.
6. Strukturbiter mot vinduer for BM25, bge-m3 og hybriden på skattekorpuset, og for BM25 med orddeling på hele korpuset.
7. Skattekorpuset mot hele korpuset for BM25 med orddeling, e5-base og hybriden med omrangering.

Bokmål og nynorsk sammenlignes bare med hvert sitt intervall, fordi spørsmålene er forskjellige. Det er mange sammenligninger, og det er ikke korrigert for det. En enkelt tydelig forskjell kan derfor skyldes tilfeldigheter.

### Kjente svakheter

Utvidelsen bruker den samme henvisningsgrafen som krysshenvisningsspørsmålene ble laget fra. Gevinsten på disse spørsmålene er derfor trolig større enn den ville vært på spørsmål fra andre kilder, og den bør leses som en øvre grense.

Alle spørsmålene er skrevet fra skattekorpuset. På hele korpuset ligger gullparagrafene fortsatt i skattelovene, men søket må finne dem blant mange flere tekster.

Utviklingssettet er lite. Valg som er gjort der, kan passe litt for godt til akkurat de spørsmålene. Derfor kommer alle tall i resultatene fra testsettet. Nynorskgruppen i testsettet har bare 27 spørsmål, så intervallene der blir brede.

De ubesvarbare spørsmålene måles først når svarene lages.

## Resultater for søk

Alle tall her kommer fra testsettet, med 198 spørsmål. Testsettet ble kjørt én gang, med valgene i planen. Tallene i hakeparentes er 95 prosent konfidensintervall. Alle tallene ligger i `results/retrieval/test.json`, og tallene for utviklingssettet ligger i `results/retrieval/dev.json`.

### Skattekorpuset med strukturbiter

| System | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| BM25 | 0,616 | 0,848 | 0,879 | 0,783 | 0,781 [0,73; 0,82] |
| BM25 med orddeling | 0,598 | 0,838 | 0,876 | 0,773 | 0,773 [0,73; 0,82] |
| e5-base | 0,631 | 0,886 | 0,934 | 0,816 | 0,823 [0,79; 0,86] |
| bge-m3 | 0,689 | 0,894 | 0,934 | 0,856 | 0,853 [0,82; 0,89] |
| nb-sbert-v2-base | 0,482 | 0,775 | 0,861 | 0,661 | 0,687 [0,64; 0,73] |
| nb-sbert-v2-large | 0,598 | 0,889 | 0,932 | 0,792 | 0,804 [0,76; 0,84] |
| Hybrid | 0,657 | 0,907 | 0,929 | 0,833 | 0,836 [0,80; 0,87] |
| Hybrid med omrangering | 0,823 | 0,944 | 0,957 | 0,948 | 0,931 [0,90; 0,96] |
| Hybrid med utvidelse | 0,657 | 0,907 | 0,942 | 0,830 | 0,843 [0,80; 0,88] |
| Hybrid med omrangering og utvidelse | 0,823 | 0,944 | 0,975 | 0,948 | 0,938 [0,91; 0,96] |

Omrangering gir det klart beste resultatet. Den løfter nDCG@10 fra 0,836 til 0,931, og MRR blir 0,948. Det betyr at en gullparagraf nesten alltid står øverst. Av modellene alene er bge-m3 best. Hybriden er klart bedre enn BM25, men den er ikke bedre enn bge-m3 alene. Den er litt svakere, men forskjellen er så liten at den kan skyldes tilfeldigheter.

### Per type spørsmål

Tabellen viser nDCG@10 på skattekorpuset med strukturbiter.

| System | Direkte (66) | Hverdagsspråk (89) | Krysshenvisning (43) |
|---|--:|--:|--:|
| BM25 | 0,891 | 0,721 | 0,737 |
| e5-base | 0,883 | 0,795 | 0,793 |
| bge-m3 | 0,835 | 0,869 | 0,849 |
| nb-sbert-v2-large | 0,827 | 0,816 | 0,742 |
| Hybrid | 0,887 | 0,806 | 0,821 |
| Hybrid med omrangering | 0,968 | 0,934 | 0,868 |
| Hybrid med omrangering og utvidelse | 0,962 | 0,938 | 0,902 |

BM25 er sterkest på direkte spørsmål, som bruker de samme juridiske ordene som loven. På hverdagsspråk er vektorsøk mye bedre, fordi spørsmålet og loven bruker forskjellige ord om det samme. Krysshenvisninger er vanskeligst, fordi begge paragrafene må finnes.

### Bokmål og nynorsk

På skattekorpuset med strukturbiter får hybriden med omrangering 0,925 [0,89; 0,95] på bokmål og 0,973 [0,93; 1,00] på nynorsk. For bge-m3 alene er tallene 0,855 og 0,840. Nynorskspørsmålene er altså ikke vanskeligere her, selv om all lovteksten er på bokmål. Gruppen har bare 27 spørsmål, og den har ingen krysshenvisninger, som er den vanskeligste typen. Det kan forklare noe av det høye tallet. Med vinduer er noen modeller svakere på nynorsk. nb-sbert-v2-large får for eksempel 0,387 mot 0,563 på bokmål.

### Planlagte sammenligninger

Tabellen viser forskjellen i nDCG@10 mellom A og B, regnet per spørsmål. En forskjell er tydelig når hele intervallet ligger på samme side av null.

| A | B | Spørsmål | Forskjell | Tydelig |
|---|---|---|--:|---|
| Hybrid, skatt | BM25, skatt | alle | +0,055 [+0,029; +0,083] | ja |
| Hybrid, skatt | bge-m3, skatt | alle | −0,017 [−0,050; +0,013] | nei |
| Hybrid, hele | BM25 med orddeling, hele | alle | +0,078 [+0,047; +0,110] | ja |
| Hybrid, hele | e5-base, hele | alle | +0,026 [−0,004; +0,056] | nei |
| Omrangering, skatt | hybrid, skatt | alle | +0,095 [+0,060; +0,128] | ja |
| Omrangering, hele | hybrid, hele | alle | +0,111 [+0,076; +0,146] | ja |
| BM25 med orddeling | BM25 | alle | −0,008 [−0,027; +0,010] | nei |
| nb-sbert-v2-base | e5-base | alle | −0,136 [−0,188; −0,090] | ja |
| nb-sbert-v2-large | bge-m3 | alle | −0,050 [−0,090; −0,006] | ja |
| Hybrid med utvidelse, skatt | hybrid, skatt | alle | +0,007 [−0,002; +0,015] | nei |
| Hybrid med utvidelse, skatt | hybrid, skatt | krysshenvisning | +0,054 [+0,028; +0,082] | ja |
| Omrangering med utvidelse, skatt | omrangering, skatt | alle | +0,007 [+0,001; +0,015] | ja |
| Omrangering med utvidelse, skatt | omrangering, skatt | krysshenvisning | +0,034 [+0,013; +0,058] | ja |
| Hybrid med utvidelse, hele | hybrid, hele | krysshenvisning | +0,017 [−0,013; +0,049] | nei |
| Omrangering med utvidelse, hele | omrangering, hele | krysshenvisning | +0,044 [+0,020; +0,070] | ja |
| BM25, strukturbiter | BM25, vinduer | alle | +0,199 [+0,158; +0,240] | ja |
| bge-m3, strukturbiter | bge-m3, vinduer | alle | +0,284 [+0,235; +0,332] | ja |
| Hybrid, strukturbiter | hybrid, vinduer | alle | +0,221 [+0,177; +0,267] | ja |
| BM25 med orddeling, strukturbiter, hele | samme, vinduer | alle | +0,214 [+0,171; +0,259] | ja |
| BM25 med orddeling, skatt | samme, hele | alle | +0,070 [+0,035; +0,107] | ja |
| e5-base, skatt | e5-base, hele | alle | +0,068 [+0,042; +0,097] | ja |
| Omrangering, skatt | omrangering, hele | alle | +0,039 [+0,015; +0,066] | ja |

Der ikke annet står, gjelder sammenligningen skattekorpuset med strukturbiter. «Omrangering» betyr hybriden med omrangering.

### Tekstbiter

Strukturbitene er klart bedre enn vinduene for alle systemene som ble sammenlignet. Forskjellen er mellom 0,20 og 0,28 i nDCG@10, og den er større enn noen annen forskjell i rapporten. Et vindu har ingen overskrift, og det kan dekke slutten av én paragraf og starten av den neste. Da kommer flere paragrafer inn i listen fra hvert treff, og den riktige paragrafen havner lenger ned. Med vinduer mister også vektorsøket forspranget sitt: bge-m3 faller fra 0,853 til 0,570, mens BM25 faller fra 0,781 til 0,582.

### Utvidelse med henvisninger

Utvidelsen hjelper på krysshenvisninger. Med omrangering stiger nDCG@10 for disse spørsmålene med 0,034 på skattekorpuset og 0,044 på hele korpuset, og recall@10 når 1,000 på skattekorpuset. Over alle spørsmål er gevinsten liten, fordi paragrafene som settes inn kan skyve riktige treff nedover på andre spørsmål. Uten omrangering faller nDCG@10 for hverdagsspråk fra 0,806 til 0,795. Gevinsten på krysshenvisninger er en øvre grense, fordi spørsmålene ble laget fra den samme henvisningsgrafen.

### Skattekorpuset og hele korpuset

Alle systemene er svakere på hele korpuset, der søket må finne de samme paragrafene blant over 50 ganger flere tekstbiter. Ordsøk og e5-base taper omtrent 0,07 i nDCG@10. Hybriden med omrangering taper bare 0,039 og ender på 0,892 [0,86; 0,92]. Hybriden på hele korpuset bruker e5-base og ikke bge-m3, så denne sammenligningen blander effekten av korpuset med effekten av modellen.

På utviklingssettet var BM25 med orddeling litt bedre enn vanlig BM25 på hele korpuset, og derfor ble den valgt til hybriden der. På testsettet var vanlig BM25 litt bedre: 0,718 mot 0,703. Orddelingen ga ingen tydelig gevinst noe sted på testsettet.

### Tid og størrelse

Vektorene ble laget på prosessoren med fire tråder. Tabellen viser tallene for strukturbitene. Indeksen lagrer hver vektor som 32-bits flyttall.

| Modell | Dimensjoner | Biter per sekund | Tid, skatt | Indeks, skatt | Tid, hele | Indeks, hele |
|---|--:|--:|--:|--:|--:|--:|
| e5-base | 768 | 3,3 | 679 s | 6,9 MB | 36 219 s | 392 MB |
| bge-m3 | 1024 | 1,0 | 2 191 s | 9,2 MB | | |
| nb-sbert-v2-base | 768 | 3,2 | 711 s | 6,9 MB | | |
| nb-sbert-v2-large | 1024 | 1,2 | 1 850 s | 9,2 MB | | |

Vinduene tar litt lengre tid per bit, fordi hvert vindu har fulle 200 ord. Tiden for bge-m3 på vinduene ble målt på nytt, fordi maskinen gikk i hvilemodus under den første målingen.

### Avvik fra planen

Alt ble kjørt slik planen sier, og alle de 22 sammenligningene ble regnet ut. Det eneste valget fra utviklingssettet som ikke holdt på testsettet, er ordsøket i hybriden for hele korpuset, som er beskrevet over.

## Plan for svar og vurdering

Denne delen ble skrevet før svarene ble laget for testsettet, og den endres ikke etterpå.

### Data og søk

Svarene lages for alle 240 spørsmålene i testsettet, også de 42 ubesvarbare. Utviklingssettet ble brukt til å prøve ut prompten og dommeren.

To søkesystemer gir tekst til generatorene: BM25 og hybriden med omrangering, begge på skattekorpuset med strukturbiter. BM25 er det enkle utgangspunktet, og hybriden med omrangering var best uten utvidelse. Utvidelse med henvisninger er ikke med, fordi gevinsten var liten og målingen på krysshenvisninger er sirkulær.

Hvert spørsmål får de fem første paragrafene fra søket som utdrag. Hvert utdrag kuttes ved 300 ord, og alle utdragene til sammen ved 1 200 ord. Da får prompten og svaret plass i de 4 096 tokenene den lokale modellen kan ta. Begge generatorene får nøyaktig samme utdrag. Antallet tokens som sendes, blir lagret for hvert svar.

### Generatorer

| Generator | Hvor den kjører | Innstillinger |
|---|---|---|
| `norallm/normistral-7b-warm-instruct`, Q4_K_M | Ollama på en bærbar PC med 4 GB skjermkort | temperatur 0,3, top-k 64, top-p 0,9, fast frø, som i modellkortet |
| `gemma-4-31b-it` | Gemini API, gratis nivå | temperatur 0 |

Filen til NorMistral oppgir 2 048 tokens som lengste kontekst, som er lengden til grunnmodellen. Modellkortet sier at instruksjonsmodellen er trent på 4 096 tokens. Derfor brukes en kopi av filen der bare dette tallet er endret til 4 096 (`scripts/set_gguf_context.py`). Vektene er de samme.

Begge får samme prompt på norsk (`src/norsk_lov_rag/generation/prompts.py`, versjon `answer-v4`). Reglene står i samme melding som utdragene og spørsmålet, fordi NorMistral er trent uten systemmeldinger.

### Svarkontrakten

Svaret skal være på samme skriftform som spørsmålet. Hver setning som sier noe om reglene, skal slutte med en kilde i fast form, for eksempel `[skatteloven § 6-44]`. Hvis utdragene ikke svarer på spørsmålet, skal modellen skrive en fast setning: «Jeg finner ikke svaret i lovtekstene jeg har fått.» på bokmål eller «Eg finn ikkje svaret i lovtekstane eg har fått.» på nynorsk. Modellene skriver svaret som vanlig tekst. Systemet bygger så det strukturerte svaret: teksten, listen over kildene i hakeparentes og et flagg som er sant når svaret er den faste setningen. Dette sjekkes mot et skjema. På utviklingssettet brøt NorMistral sammen når den ble tvunget til å skrive JSON, og derfor skriver begge modellene vanlig tekst.

### Kontroller uten modell

En kilde gjøres om til en paragraf ved å lese navnet på loven og paragrafnummeret, uansett mellomrom, tankestrek og store bokstaver. Nynorske navn som «skattelova» leses som «skatteloven». Deretter sjekkes det om svaret er gyldig JSON, om hver kilde finnes og var blant utdragene modellen fikk, om hver setning har en kilde, og om svaret er skrevet på riktig skriftform. Skriftformen sjekkes med de samme kjenneordene som i filteret for testsettet.

### Dommer

Dommeren er `gemini-3.6-flash` med temperatur 0. Den vurderer to ting med en skriftlig rubrikk på norsk. Riktighet vurderes mot referansesvaret som riktig, delvis eller feil. Støtte vurderes for hver kilde: sier teksten i paragrafen det setningene med denne kilden sier? En kilde som ikke peker på noen kjent paragraf, regnes som uten støtte og sendes ikke til dommeren.

Svar fra alle systemene blandes i en fast tilfeldig rekkefølge og sendes 50 om gangen, med en kort kode i stedet for navnet på systemet. Dommeren ser aldri hvilket system som skrev et svar. Et tilfeldig utvalg på 50 svar vurderes en gang til, i ny rekkefølge og nye grupper. Samsvaret mellom de to vurderingene måles som andel like vurderinger og som Cohens kappa.

### Mål

| Mål | Hvordan det regnes |
|---|---|
| Riktighet | Riktig teller 1, delvis 0,5 og feil 0. Å avstå på et spørsmål som har svar, teller 0. Snitt over spørsmålene med svar, og andelen som er helt riktige. |
| Presisjon for kilder | Andelen av kildene i et svar som er gullparagrafer. Regnes for svar som har minst én kilde. |
| Dekning for kilder | Andelen av gullparagrafene som svaret viser til. Regnes for alle spørsmål med svar, så et svar der modellen avstår, gir 0. |
| Mild telling av kilder | Presisjon og dekning når alle paragrafer som er nevnt i teksten telles, også uten hakeparentes. Dette er et tilleggsmål. |
| Støtte | Andelen av kildene i et svar som dommeren mener støtter påstanden. |
| Avstå | Riktige avslag på ubesvarbare spørsmål, og feil avslag på spørsmål som har svar. |
| Skriftform | Andelen svar som ikke er skrevet på feil skriftform. |
| Tokens og tid | Tokens per spørsmål, og median og 95-persentil for svartiden. |

Hvert snitt får et 95 prosent konfidensintervall fra bootstrap over spørsmålene. Resultatene vises også per type spørsmål og per skriftform. Svartiden måles ulikt for de to generatorene: lokalt avhenger den av den bærbare PC-en, og gjennom API-et av belastningen hos Google.

### Sammenligninger

Hovedmålet er riktighet. Disse sammenligningene er bestemt på forhånd, med parvise forskjeller over de samme spørsmålene:

1. Gemma mot NorMistral med hybriden med omrangering.
2. Gemma mot NorMistral med BM25.
3. Hybriden med omrangering mot BM25 for Gemma.
4. Hybriden med omrangering mot BM25 for NorMistral.

En forskjell regnes som tydelig når hele intervallet ligger på samme side av null.

### Kjente svakheter

Dommeren og Gemma kommer begge fra Google. Det kan gi Gemma en fordel hos dommeren. Dommeren er en språkmodell, og ingen mennesker kontrollerer vurderingene. Referansesvarene ble skrevet av `openai/gpt-oss-120b`, så riktighet måles mot dem. Den gratis kvoten til dommeren er 20 kall per dag, og derfor sendes 50 svar i hvert kall. Lange grupper kan gjøre vurderingene litt dårligere. Kontrollen av samsvar måler hvor stabil dommeren er, ikke om den har rett.

NorMistral trekker ord med en liten tilfeldighet. Med fast frø gir den samme svar på samme maskin, men ikke nødvendigvis på en annen. En paragraf som er lengre enn 300 ord, blir kuttet, og da kan den delen som svarer på spørsmålet, falle bort.

## Resultater for svar og vurdering

Alle 240 spørsmålene i testsettet fikk svar fra alle fire kombinasjonene av søk og generator, til sammen 960 svar. Ingen spørsmål falt bort. Dommeren vurderte alle de 753 svarene som skulle vurderes. Tallene i hakeparentes er 95 prosent konfidensintervall. Alle tallene og feileksemplene ligger i `results/generation/test.json`.

### Hovedtall

Riktighet og dekning gjelder de 198 spørsmålene som har svar. Presisjon og støtte gjelder svarene som har minst én kilde i hakeparentes.

| Generator og søk | Riktighet | Helt riktig | Presisjon kilder | Dekning kilder | Støtte |
|---|--:|--:|--:|--:|--:|
| NorMistral, BM25 | 0,631 [0,57; 0,69] | 0,525 | 0,424 [0,24; 0,60] | 0,056 [0,03; 0,09] | 0,405 [0,25; 0,57] |
| NorMistral, hybrid med omrangering | 0,641 [0,58; 0,70] | 0,556 | 0,362 [0,22; 0,51] | 0,068 [0,04; 0,11] | 0,574 [0,41; 0,74] |
| Gemma, BM25 | 0,821 [0,77; 0,87] | 0,803 | 0,861 [0,82; 0,90] | 0,768 [0,71; 0,82] | 0,988 [0,97; 1,00] |
| Gemma, hybrid med omrangering | 0,917 [0,88; 0,95] | 0,899 | 0,825 [0,79; 0,87] | 0,869 [0,83; 0,91] | 0,984 [0,96; 1,00] |

Gemma med hybriden med omrangering er best. Dommeren mener at 90 prosent av svarene på spørsmål som har svar, er helt riktige. Gemma er klart bedre enn NorMistral med begge søkesystemene. Bedre søk hjelper Gemma, men ikke NorMistral.

### Kilder og avslag

| Generator og søk | Mild presisjon | Mild dekning | Riktige avslag | Feil avslag | Riktig skriftform | Alle setninger med kilde | Kuttet |
|---|--:|--:|--:|--:|--:|--:|--:|
| NorMistral, BM25 | 0,341 | 0,260 | 0,000 [0,00; 0,00] | 0,010 [0,00; 0,03] | 0,983 | 0,000 | 33 |
| NorMistral, hybrid med omrangering | 0,349 | 0,260 | 0,024 [0,00; 0,07] | 0,020 [0,01; 0,04] | 0,975 | 0,004 | 28 |
| Gemma, BM25 | 0,812 | 0,780 | 0,976 [0,93; 1,00] | 0,157 [0,11; 0,21] | 0,996 | 0,958 | 1 |
| Gemma, hybrid med omrangering | 0,769 | 0,874 | 0,976 [0,93; 1,00] | 0,051 [0,02; 0,08] | 0,996 | 0,963 | 1 |

Gemma følger svarkontrakten. Over 95 prosent av svarene som ikke er avslag, har en kilde etter hver setning. Ingen kilde peker utenfor utdragene, og dommeren mener at i snitt 98 prosent av kildene støtter påstanden. Gemma avstår riktig på 41 av 42 ubesvarbare spørsmål. Med BM25 avstår den også på 16 prosent av spørsmålene som har svar. Med hybriden med omrangering faller dette til 5 prosent, trolig fordi utdragene oftere inneholder svaret.

NorMistral følger ikke svarkontrakten. Den skriver flytende svar, men setter nesten aldri kilden i hakeparentes. Bare 31 og 27 av de 240 svarene har en slik kilde. Den milde tellingen, som også teller paragrafer som er nevnt i teksten, gir en dekning på 0,26. Av svarene med kilde i hakeparentes viser 18 og 24 prosent til en paragraf som ikke var blant utdragene. NorMistral avstår nesten aldri. Den svarer på 42 og 41 av de 42 ubesvarbare spørsmålene, for eksempel om skattesatsen for i år, uten at lovteksten sier det. Mellom 12 og 14 prosent av svarene ble kuttet ved grensen på 512 tokens.

Begge generatorene skriver på riktig skriftform i over 97 prosent av svarene.

### Per type spørsmål

Tabellen viser riktighet. For ubesvarbare spørsmål viser den andelen riktige avslag.

| Generator og søk | Krysshenvisning (43) | Direkte (66) | Hverdagsspråk (89) | Ubesvarbar (42) |
|---|--:|--:|--:|--:|
| NorMistral, BM25 | 0,523 | 0,765 | 0,584 | 0,000 |
| NorMistral, hybrid med omrangering | 0,547 | 0,689 | 0,652 | 0,024 |
| Gemma, BM25 | 0,674 | 0,970 | 0,781 | 0,976 |
| Gemma, hybrid med omrangering | 0,802 | 0,962 | 0,938 | 0,976 |

Spørsmål på hverdagsspråk har mest å vinne på bedre søk. For Gemma stiger riktigheten der fra 0,781 til 0,938. Krysshenvisninger er vanskeligst for begge generatorene.

### Bokmål og nynorsk

| Generator og søk | Bokmål (207) | Nynorsk (33) |
|---|--:|--:|
| NorMistral, BM25 | 0,632 | 0,630 |
| NorMistral, hybrid med omrangering | 0,635 | 0,685 |
| Gemma, BM25 | 0,827 | 0,778 |
| Gemma, hybrid med omrangering | 0,921 | 0,889 |

Forskjellene mellom bokmål og nynorsk er små. Det er bare 33 spørsmål på nynorsk, så tallene er usikre.

### Tokens og tid

| Generator og søk | Tokens inn | Tokens ut | Median tid | 95-persentil |
|---|--:|--:|--:|--:|
| NorMistral, BM25 | 1 391 | 266 | 51,0 s | 97,3 s |
| NorMistral, hybrid med omrangering | 1 518 | 268 | 49,7 s | 99,8 s |
| Gemma, BM25 | 1 823 | 1 163 | 53,4 s | 116,1 s |
| Gemma, hybrid med omrangering | 1 994 | 1 202 | 55,5 s | 103,0 s |

Tokens ut for Gemma tar med tenkingen før svaret. Selve svaret er kort. Tiden er målt for kall som lyktes. Ventetid etter serverfeil er ikke med.

### Planlagte sammenligninger

Tabellen viser forskjellen i riktighet mellom A og B, regnet per spørsmål over de 198 spørsmålene som har svar.

| A | B | Forskjell i riktighet | Tydelig |
|---|---|--:|---|
| Gemma, hybrid med omrangering | NorMistral, hybrid med omrangering | +0,275 [+0,215; +0,338] | ja |
| Gemma, BM25 | NorMistral, BM25 | +0,189 [+0,121; +0,258] | ja |
| Gemma, hybrid med omrangering | Gemma, BM25 | +0,096 [+0,048; +0,146] | ja |
| NorMistral, hybrid med omrangering | NorMistral, BM25 | +0,010 [−0,053; +0,073] | nei |

Tre av de fire forskjellene er tydelige. Bedre søk gir ikke bedre svar fra NorMistral. Den bruker utdragene for dårlig til at bedre utdrag hjelper.

### Dommeren

Dommeren vurderte 50 svar to ganger. Riktigheten var lik i 40 av 50 tilfeller (80 prosent), og Cohens kappa er 0,54. Støtten var lik for 40 av 41 kilder. Dommeren er altså stabil på støtte, men bare middels stabil på riktighet. Forskjeller i riktighet på noen få hundredeler bør derfor leses med forsiktighet. De tydelige forskjellene over er større enn dette.

### Feileksempler

Tabellen teller svar med hver type feil, over alle fire kombinasjonene. Et svar kan ha flere typer feil. Eksempler på hver type ligger i `results/generation/test.json` og på siden «Resultater» i demoen.

| Feil | Antall | Mest fra |
|---|--:|---|
| Feil svar på spørsmål som har svar | 104 | NorMistral (100) |
| Svar i stedet for avslag på ubesvarbart spørsmål | 85 | NorMistral (83) |
| Avslag på spørsmål som har svar | 47 | Gemma med BM25 (31) |
| Kilde uten støtte | 30 | NorMistral (25) |
| Kilde utenfor utdragene | 17 | NorMistral (17) |
| Feil skriftform | 12 | begge |
| Ugyldig svar | 5 | begge |

Et typisk feil svar fra NorMistral sier det motsatte av loven, for eksempel at inngående merverdiavgift ikke skal tilbakeføres i et tilfelle der loven krever det. Med BM25 gjelder de fleste unødvendige avslagene fra Gemma spørsmål på hverdagsspråk (18 av 31), der ordsøket treffer dårligst. Et av Gemmas få feil svar var tomt: modellen brukte hele grensen på tenking.

### Avvik fra planen

Dommeren fikk først grupper på 50 svar og en grense på 16 000 tokens. Dommeren tenker før den svarer. I det andre kallet gikk nesten hele grensen til tenking, og bare noen få vurderinger kom tilbake. Med 50 svar per gruppe ville den daglige kvoten på 20 kall ikke holdt. Derfor ble gruppene økt til 80 svar og grensen til 60 000 tokens, og alle svarene ble vurdert på nytt med disse innstillingene. Vurderingene fra de to første kallene er ikke brukt. Det ble også lagt til en regel om å sende svar som dommeren hoppet over, en gang til. Den trengtes ikke, fordi dommeren vurderte alle svarene. Alt annet ble kjørt slik planen sier.

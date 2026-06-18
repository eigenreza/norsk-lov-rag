# Metode

Denne filen beskriver hvordan systemet er bygd: fra lovteksten blir lastet ned, til svarene blir vurdert. Hvert steg er en kommando i `norsk-lov-rag`, og alle innstillinger står i `configs/default.yaml`. Planen for evalueringen og resultatene står i `valideringsrapport.md`.

## Korpus

Lovteksten kommer fra Lovdatas åpne datasett, hentet gjennom det offentlige API-et uten nøkkel. Prosjektet bruker to arkiver: gjeldende lover og gjeldende sentrale forskrifter. Filene lagres slik de kommer, og de identifiseres med SHA-256. Korpuset er merket med en hash av innholdet, ikke med en dato. Se `datakilder.md` for vilkårene og hashen.

Hvert dokument er HTML som også er gyldig XML. Parseren leser strukturen i loven: kapittel, paragraf, ledd, bokstav og nummererte punkter. Endringsnotater, fotnoter og paragrafer som ikke har trådt i kraft ennå, blir holdt utenfor teksten. Hver rad får en id i samme form som lenkene til Lovdata, for eksempel `lov/1999-03-26-14/§6-44/ledd/1`. Opphevede og tomme paragrafer blir merket og er ikke med i søket.

Lovdata merker mange dokumenter som bokmål eller nynorsk, men de fleste forskriftene er bare merket som norsk. For disse teller systemet ord som bare finnes i den ene skriftformen. Metoden er kontrollert mot dokumentene som har en tydelig merking.

Arkivene har 5 909 dokumenter. Stortingsvedtak, instrukser og delegeringer er ikke med. Hele korpuset består derfor av 753 lover og 3 457 sentrale forskrifter, til sammen 4 210 dokumenter med 107 534 gjeldende paragrafer. 227 av dokumentene er på nynorsk. Stortingets årlige skattevedtak ligger blant stortingsvedtakene. Derfor har korpuset ikke satsene og beløpsgrensene for et gitt år.

Skattekorpuset har ni dokumenter med 2 023 gjeldende paragrafer. Alle er skrevet på bokmål.

| Dokument | Nøkkel |
|---|---|
| skatteloven | `lov/1999-03-26-14` |
| skatteforvaltningsloven | `lov/2016-05-27-14` |
| skatteforvaltningsforskriften | `forskrift/2016-11-23-1360` |
| Finansdepartementets skattelovforskrift (FSFIN) | `forskrift/1999-11-19-1158` |
| Skattedirektoratets skatteforskrift (FSSD) | `forskrift/1999-11-22-1160` |
| merverdiavgiftsloven | `lov/2009-06-19-58` |
| merverdiavgiftsforskriften | `forskrift/2009-12-15-1540` |
| skattebetalingsloven | `lov/2005-06-17-67` |
| skattebetalingsforskriften | `forskrift/2007-12-21-1766` |

Etter parsingen kjøres faste kontroller av dataene. De sjekker blant annet at alle id-er er unike, at ingen gjeldende paragraf er tom, og hvor stor andel som er på nynorsk.

## Henvisninger

Lovtekst viser ofte til andre bestemmelser, for eksempel «jf. § 5-1», «§§ 5-10 til 5-15» eller «skatteloven § 14-4 første ledd». En egen parser finner slike henvisninger i teksten. Deretter finner systemet målet. Det leser først hvilket dokument henvisningen gjelder: loven som er nevnt ved navn, dokumentet henvisningen står i, eller loven en forskrift bygger på når teksten sier «loven». Så slår det opp paragrafen, og leddet når det er oppgitt.

Henvisningene lagres som en graf med én kant per henvisning. I skattekorpuset blir 93,2 prosent av henvisningene til en paragraf knyttet til riktig mål. I hele korpuset er tallet 89,1 prosent. Den vanligste grunnen til at en henvisning ikke blir knyttet til noe, er at paragrafen ikke blir funnet i dokumentet det er vist til.

Lovdatas redaktører har selv lagt lenker inn i teksten. De brukes ikke til å bygge grafen, bare til å måle den. I skattekorpuset finnes 96,3 prosent av parene i grafen også blant Lovdatas lenker, og grafen har med 95,8 prosent av lenkene. I hele korpuset er tallene 91,8 og 88,8 prosent.

Grafen brukes til to ting: til å lage spørsmål om krysshenvisninger i testsettet, og til å utvide søkeresultatene med paragrafene et treff viser til.

## Tekstbiter

Teksten deles opp på to måter, slik at virkningen kan måles. Strukturbiter følger loven. Hver bit er én paragraf, med lovens navn, kapittel og paragraftittel foran teksten. En paragraf med mer enn 300 ord deles ved grensene mellom leddene. Vinduer ser bort fra strukturen. Hvert vindu har 200 ord, og et nytt vindu starter hvert 150. ord.

Et søk gir en liste med tekstbiter, men det som måles, er en liste med paragrafer. Hver bit vet derfor hvilke paragrafer den dekker. Et vindu som dekker flere paragrafer, teller først for paragrafen med flest ord i vinduet.

## Søk

Ordsøket er BM25 med små bokstaver, norske stoppord og Snowball-stammeren for norsk. Norsk skriver sammensatte ord i ett, og da deler for eksempel «merverdiavgiftsregisteret» ingen ord med «registeret». Derfor finnes også en variant som deler lange ord i to kjente ord, med s eller e mellom. Et ord regnes som kjent når det finnes ofte nok i korpuset selv.

Vektorsøket bruker fire modeller: to flerspråklige (`intfloat/multilingual-e5-base` og `BAAI/bge-m3`) og to norske (`NbAiLab/nb-sbert-v2-base` og `NbAiLab/nb-sbert-v2-large`). Alle kjører på prosessoren. Vektorene er normalisert og sammenlignes med skalarprodukt. På hele korpuset er bare e5-base kjørt, fordi de større modellene ville trengt mer enn et døgn hver.

Hybriden slår sammen det beste ordsøket og det beste vektorsøket med resiprok rangfusjon (k = 60). Valget ble gjort på utviklingssettet. På skattekorpuset er det BM25 og bge-m3, og på hele korpuset BM25 med orddeling og e5-base. Til slutt kan `BAAI/bge-reranker-v2-m3` ordne de 30 første tekstbitene fra hybriden på nytt. Utvidelsen setter inn paragrafene som de fem første treffene viser til, rett etter de fem treffene.

## Svar

To søkesystemer gir tekst til generatorene: BM25 og hybriden med omrangering, begge på skattekorpuset med strukturbiter. Hvert spørsmål får de fem første paragrafene som utdrag. Hvert utdrag kuttes ved 300 ord, og alle til sammen ved 1 200 ord. Hvert utdrag har en fast merkelapp, for eksempel `[skatteloven § 6-44]`, og tittelen på paragrafen.

Prompten er på norsk og står i én melding sammen med utdragene og spørsmålet. Den har seks regler. Modellen skal svare bare ut fra utdragene og på samme skriftform som spørsmålet. Hver setning om reglene skal slutte med merkelappen til kilden i hakeparentes. Når utdragene ikke svarer på spørsmålet, skal modellen skrive en fast setning: «Jeg finner ikke svaret i lovtekstene jeg har fått.» eller den samme setningen på nynorsk. Til slutt gjentar en kort påminnelse de viktigste reglene. Prompten har versjonen `answer-v4`.

Modellene skriver vanlig tekst. Systemet bygger så det strukturerte svaret: teksten, kildene i hakeparentes og et flagg for avslag. Svaret sjekkes mot et skjema. Tekst med rester av chat-formatet eller JSON regnes som ugyldig, slik at feil i formatet blir talt og ikke skjult. Den lokale modellen brøt sammen når den ble tvunget til å skrive JSON. Derfor skriver begge modellene vanlig tekst.

Den første generatoren er `norallm/normistral-7b-warm-instruct`. Den kjører lokalt i Ollama som en kvantisert fil med omtrent fire biter per vekt (Q4_K_M), på en bærbar PC med 4 GB skjermkort. Innstillingene følger modellkortet: temperatur 0,3, top-k 64, top-p 0,9 og fast frø. Filen oppgir 2 048 tokens som lengste kontekst, som er lengden til grunnmodellen. Instruksjonsmodellen er trent på 4 096 tokens, og Ollama ville ellers kuttet lange prompter uten å si fra. Derfor brukes en kopi av filen der bare dette tallet er endret (`scripts/set_gguf_context.py`). Vektene er de samme.

Den andre generatoren er `gemma-4-31b-it` gjennom Gemini API på gratis nivå, med temperatur 0. Gemma tenker før den svarer, og tenkingen teller med i grensen for lengden på svaret. Grensen er derfor 8 192 tokens for Gemma og 512 tokens for NorMistral. Gratis nivå svarer ofte med en serverfeil. Klienten venter og prøver igjen opptil åtte ganger. Et spørsmål som fortsatt feiler, blir hoppet over og fylt inn ved neste kjøring.

## Kontroll av kilder

En kilde gjøres om til en paragraf ved å lese navnet på loven og paragrafnummeret, uansett mellomrom, bindestrek og store bokstaver. Nynorske navn som «skattelova» leses som «skatteloven», og forkortelsene FSFIN og FSSD gjenkjennes. En kilde som ikke peker på noen kjent paragraf, beholdes som uløst.

For hvert svar sjekker systemet om hver kilde finnes, om den var blant utdragene, og om hver setning har en kilde. Skriftformen sjekkes med de samme kjenneordene som i filteret for testsettet. Noen svar nevner en paragraf i teksten uten hakeparentes. Et tilleggsmål teller også disse.

## Dommer

Riktighet og støtte vurderes av `gemini-3.6-flash` med temperatur 0 og en skriftlig rubrikk på norsk. Riktighet vurderes mot referansesvaret som riktig, delvis eller feil. Støtte vurderes for hver kilde: dommeren får setningene som viser til kilden og teksten i paragrafen, og svarer om teksten sier det setningene sier.

Avslag sendes ikke til dommeren. Det gjør heller ikke en kilde som ikke peker på noen kjent paragraf. Den regnes som uten støtte. Svar fra alle systemene blandes i en fast tilfeldig rekkefølge og sendes 50 om gangen. Hvert svar har en kort kode i stedet for navnet på systemet, så dommeren vet ikke hvem som skrev det. Et tilfeldig utvalg på 50 svar vurderes en gang til, i ny rekkefølge og nye grupper. Samsvaret måles som andel like vurderinger og som Cohens kappa.

## Mål og usikkerhet

Søket måles med recall@1, recall@5, recall@10, MRR og nDCG@10. Svarene måles med riktighet, presisjon og dekning for kilder, støtte, riktige og feil avslag, skriftform, tokens og svartid. Den nøyaktige definisjonen av hvert mål står i `valideringsrapport.md`.

Hvert snitt får et 95 prosent konfidensintervall fra bootstrap. Spørsmålene trekkes på nytt med tilbakelegging 2 000 ganger med fast frø, og intervallet er 2,5- og 97,5-persentilen av snittene. To systemer sammenlignes med forskjellen per spørsmål, der de samme spørsmålene trekkes for begge.

## Hurtigbuffer og gratis kvoter

Alle kall til språkmodeller går gjennom én klient. Hvert svar lagres med en nøkkel laget av modellen, versjonen av prompten, meldingene og innstillingene. En ny kjøring finner svarene der og gjør ingen nye kall, så hvert steg kan kjøres på nytt uten nett. Klienten kjenner grensene for hver gratis plan. Den venter når en minuttgrense er nådd, og stopper når en dagsgrense er nådd. Den går aldri over til en annen modell eller en betalt tjeneste.

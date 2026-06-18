# Modellkort

Dette kortet beskriver hva systemet er laget for, hva det bygger på, hvor godt det fungerer og hvor det ikke bør brukes.

## Formål

`norsk-lov-rag` svarer på spørsmål om norsk lovtekst, på norsk. Systemet finner de relevante paragrafene i lover og forskrifter og lager et kort svar ut fra dem. Hver setning i svaret viser til paragrafen den bygger på. Når lovteksten ikke svarer på spørsmålet, sier systemet det i stedet for å gjette. Metoden virker på alle norske lover og forskrifter. Skatteretten er brukt til evalueringen og eksemplene.

## Tiltenkt bruk

Systemet er laget for forskning og demonstrasjon. Det viser hvordan norsk lovtekst kan behandles, hvordan søkemetoder for norsk kan sammenlignes på en rettferdig måte, og hvordan svar med kontrollerte kilder kan måles.

Systemet gir ikke juridiske råd eller skatteråd. Svarene lages av en språkmodell og kan være feil. Lovteksten er et fast uttrekk som kan ha endret seg siden. Prosjektet har ingen tilknytning til Skatteetaten, Lovdata eller andre offentlige organer. Den som trenger hjelp med en konkret sak, bør spørre Skatteetaten eller en advokat.

## Data

Lovteksten kommer fra Lovdatas åpne datasett med gjeldende lover og sentrale forskrifter, under Norsk lisens for offentlige data (NLOD) 2.0. Skattekorpuset har ni lover og forskrifter med 2 023 gjeldende paragrafer. Hele korpuset har 4 210 dokumenter. Korpuset er merket med en hash av innholdet. Se `datakilder.md`.

Testsettet har 344 spørsmål på bokmål og nynorsk, delt i 104 til utvikling og 240 til test. Spørsmålene ble skrevet, kontrollert og gjennomgått av språkmodeller. Se `testsett.md`.

## Modeller

| Rolle | Modell |
|---|---|
| Ordsøk | BM25 med norsk ordstamming |
| Vektorsøk | `BAAI/bge-m3` på skattekorpuset, `intfloat/multilingual-e5-base` på hele korpuset |
| Omrangering | `BAAI/bge-reranker-v2-m3` |
| Svar, lokalt | `norallm/normistral-7b-warm-instruct`, kvantisert til fire biter i Ollama |
| Svar, API | `gemma-4-31b-it` gjennom Gemini API, gratis nivå |
| Dommer | `gemini-3.6-flash` gjennom Gemini API, gratis nivå |

Lisensene står i `datakilder.md`. Metoden er beskrevet i `metode.md`.

## Resultater

Alle tall kommer fra testsettet. Tallene i hakeparentes er 95 prosent konfidensintervall.

Søket er godt. På skattekorpuset gir hybriden med omrangering nDCG@10 på 0,931 [0,90; 0,96] og recall@10 på 0,957. En riktig paragraf står nesten alltid øverst (MRR 0,948). Til sammenligning får BM25 alene 0,781 og bge-m3 alene 0,853. På hele korpuset, der de samme paragrafene må finnes blant over 50 ganger flere tekstbiter, får hybriden med omrangering 0,892 [0,86; 0,92]. Strukturbiter som følger loven, er klart bedre enn tekstvinduer med fast lengde.

Svarene er gode med Gemma og svake med NorMistral. Gemma med hybriden med omrangering har riktighet 0,917 [0,88; 0,95] på spørsmålene som har svar, kildene støtter påstanden i 98 prosent av tilfellene, og den avstår riktig på 41 av 42 ubesvarbare spørsmål. NorMistral har riktighet 0,64, setter sjelden kilder i fast form og avstår nesten aldri.

Resultatene for svarene, med riktighet, kilder, støtte og avslag for hver kombinasjon av søk og generator, står i `valideringsrapport.md`.

## Begrensninger

Ingen mennesker har lest testspørsmålene eller kontrollert vurderingene til dommeren. Dommeren og Gemma kommer begge fra Google, og det kan gi Gemma en fordel. Testsettet er lite, særlig for nynorsk og krysshenvisninger, så intervallene der er brede. Lovteksten er et fast uttrekk, og korpuset har ikke de årlige skattevedtakene med satser og beløpsgrenser.

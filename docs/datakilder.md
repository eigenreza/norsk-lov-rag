# Datakilder, tjenester og lisenser

Denne filen viser hvor dataene kommer fra, hvilke modeller prosjektet bruker, og vilkårene for
hver av dem. En modell blir ikke brukt før lisensen er lest og ført opp her.

## Lovtekst fra Lovdata

Lovteksten kommer fra Lovdatas åpne datasett, hentet gjennom det offentlige API-et uten nøkkel
(`https://api.lovdata.no/v1/publicData`). Prosjektet bruker to datasett:

| Datasett | Innhold |
|---|---|
| `gjeldende-lover.tar.bz2` | gjeldende lover |
| `gjeldende-sentrale-forskrifter.tar.bz2` | gjeldende sentrale forskrifter |

Dataene er lisensiert under Norsk lisens for offentlige data (NLOD) 2.0. Kildeangivelsen er:

> Inneholder data under Norsk lisens for offentlige data (NLOD) tilgjengeliggjort av Stiftelsen
> Lovdata.

Lovdata oppdaterer datasettene hver natt. Korpuset er derfor merket med en hash av innholdet,
ikke med en dato. Resultatene i prosjektet bygger på korpuset med denne hashen:

`829b4b28d31c142a30bf1cdf7c8d6e353c9eb91d9d67f104f1ad5aafb9690772`

## Språkmodeller på gratis planer

Alle kall går gjennom én klient (`src/norsk_lov_rag/llm`). Klienten kjenner grensene for hver
gratis plan, venter når en minuttgrense er nådd, og stopper når en dagsgrense er nådd. Den går
aldri over til en annen modell eller en betalt tjeneste. Nøklene hører til prosjekter uten
betalingsinformasjon, så ingen kall kan koste penger.

### Groq, gratis plan

- Ingen betalingsinformasjon trengs. Et betalingskort blir bare bedt om ved oppgradering til en
  betalt plan, og det er ikke gjort.
- Grenser per modell, lest fra svarene fra API-et: 1 000 forespørsler per dag og 8 000 tokens
  per minutt. Groq oppgir også 30 forespørsler per minutt og 200 000 tokens per dag for
  `openai/gpt-oss-120b`. Over grensen svarer API-et med feilkode 429.
- Bruk av data: Groq lagrer ikke innholdet i forespørslene som standard. Logger kan bli
  oppbevart i inntil 30 dager for å hindre misbruk og holde tjenesten stabil.
- Groq teller hele `max_tokens` mot minuttgrensen når forespørselen sendes. Klienten regner
  derfor med det samme når den planlegger kall.
- Svarene fra API-et viser at grensene fylles på jevnt over tid. Klienten følger samme regel.

### Google Gemini API, gratis nivå

- Nøkkelen hører til et eget Google Cloud-prosjekt uten fakturering.
- Norge er blant landene der API-et kan brukes. For brukere i EØS gjelder vilkårene for betalt
  bruk også på gratis nivå. Det betyr at Google ikke bruker innholdet til å forbedre sine
  produkter.
- Google viser grensene for gratis nivå bare i AI Studio. For `gemini-3.6-flash` svarte API-et
  med feilkode 429 for dagsgrensen etter 20 forsøk regnet fra midnatt i California, og da var
  også mislykkede kall med i tallet. Klienten bruker derfor 5 forespørsler per minutt og 20 per
  dag, teller mislykkede kall og starter dagen på nytt ved midnatt i California.

## Modeller

| Rolle | Modell | Tjeneste | Lisens |
|---|---|---|---|
| Skriver testspørsmål | `openai/gpt-oss-120b` | Groq | Apache-2.0 (åpne vekter) |
| Kontrollerer at spørsmål kan besvares | `qwen/qwen3.8-27b` | Groq | Apache-2.0 (åpne vekter) |
| Lager søkeord og svar i demoen | `openai/gpt-oss-120b` | Groq | Apache-2.0 (åpne vekter) |
| Går gjennom testsettet | `gemini-3.6-flash` | Google Gemini API | lukket modell, Gemini API-vilkår |
| Svarer på spørsmål | `norallm/normistral-7b-warm-instruct` | Ollama på egen maskin | Apache-2.0 |
| Svarer på spørsmål | `gemma-4-31b-it` | Google Gemini API | Apache-2.0 (åpne vekter) |
| Vurderer svarene | `gemini-3.6-flash` | Google Gemini API | lukket modell, Gemini API-vilkår |

Modellen som skriver spørsmålene, er fra en annen modellfamilie enn modellene som blir testet
som generatorer. Groq merker `qwen/qwen3.8-27b` som en forhåndsversjon. Versjonen som svarer,
blir lagret sammen med hvert svar i hurtigbufferen.

## Modeller for søk

Modellene for søk kjører lokalt på maskinens prosessor. De lastes ned fra Hugging Face i en fast
versjon og sendes ikke til noen tjeneste. Ingen av dem krever at man godtar egne vilkår før
nedlasting.

| Rolle | Modell | Versjon | Lisens |
|---|---|---|---|
| Flerspråklig vektorsøk | `intfloat/multilingual-e5-base` | `d1287505` | MIT |
| Flerspråklig vektorsøk | `BAAI/bge-m3` | `5617a9f6` | MIT |
| Norsk vektorsøk | `NbAiLab/nb-sbert-v2-base` | `e3392202` | Apache-2.0 |
| Norsk vektorsøk | `NbAiLab/nb-sbert-v2-large` | `b7377067` | Apache-2.0 |
| Omrangering | `BAAI/bge-reranker-v2-m3` | `953dc6f6` | Apache-2.0 |

Hele versjonsnummeret står i `configs/default.yaml`. `NbAiLab/borealis-embed-212m` er ikke brukt.
Den bygger på Gemma, og Googles vilkår for Gemma gjelder i tillegg til Nasjonalbibliotekets egen
lisens.

## Modeller for svar

NorMistral kjører lokalt i Ollama, fra filen `normistral-7b-warm-instruct.Q4_K_M.gguf` i
modellens eget repo på Hugging Face (`ltg/normistral-7b-warm-instruct`, versjon `79abf428`). Filen
er en kvantisert utgave med fire biter per vekt, slik at modellen passer på en bærbar PC.
Modellen er lisensiert under Apache-2.0 og krever ikke at man godtar egne vilkår. Modellkortet
sier at instruksjonsdataene kommer fra åpne datasett. Ett av dem, No Robots, har en lisens som
ikke tillater kommersiell bruk, og Dolly har en lisens som krever deling på samme vilkår.
Utviklerne har likevel gitt ut modellen under Apache-2.0. Prosjektet bruker den bare til
forskning og demonstrasjon.

Gemma 4 er gitt ut av Google under Apache-2.0. Her brukes den gjennom Gemini API på gratis
nivå, med modellnavnet `gemma-4-31b-it`. Google oppgir ikke grensene for gratis nivå utenfor AI
Studio, så klienten bruker lave grenser og stopper ved feilkode 429 for dagsgrensen.

Dommeren er den samme modellen som gikk gjennom testsettet, med de samme grensene. Den er en
nyere modell enn begge generatorene, og den får ikke vite hvilket system som skrev et svar.

# norsk-lov-rag

**Interaktiv demo:** [norsk-lov-rag.azurewebsites.net](https://norsk-lov-rag.azurewebsites.net). Still et spørsmål om skatterett og prøv systemet selv.

Spørsmål og svar om norsk lovtekst, med kilder til hver påstand. Systemet er en RAG for norsk lovtekst, evaluert på skatterett.

[English version](README.en.md)

> **Viktig:** Svarene lages av en språkmodell og kan være feil. De er ikke juridiske råd eller skatteråd. Lovteksten er et fast uttrekk og kan ha endret seg. Prosjektet har ingen tilknytning til Skatteetaten, Lovdata eller andre offentlige organer.

## Hva systemet gjør

Brukeren stiller et spørsmål på bokmål eller nynorsk. Systemet finner de relevante paragrafene i norske lover og forskrifter og svarer på norsk. Hver setning i svaret viser til paragrafen den bygger på, for eksempel `[skatteloven § 6-44]`. Når lovteksten ikke svarer på spørsmålet, sier systemet det i stedet for å gjette.

Prosjektet består av disse delene:

- en parser for strukturen i Lovdatas dokumenter, fra kapittel til bokstav, og en graf over henvisninger som «jf. § 5-1»;
- en sammenligning av ordsøk, flerspråklige og norske vektormodeller, hybridsøk og omrangering, på skattekorpuset og på alle gjeldende lover og forskrifter;
- svar fra en lokal norsk modell og en modell gjennom et gratis API, med kontroll av kildene og avslag når teksten ikke svarer;
- et testsett med 344 spørsmål og en evaluering med konfidensintervaller.

Metoden virker på alle norske lover og forskrifter. Skatteretten er brukt til evalueringen og eksemplene.

## Resultater

Søket er målt på 198 spørsmål i testsettet som har svar i korpuset. Tallene i hakeparentes er 95 prosent konfidensintervall.

| System | Korpus | Recall@10 | MRR | nDCG@10 |
|---|---|--:|--:|--:|
| BM25 | skatt | 0,879 | 0,783 | 0,781 [0,73; 0,82] |
| bge-m3 | skatt | 0,934 | 0,856 | 0,853 [0,82; 0,89] |
| Hybrid med omrangering | skatt | 0,957 | 0,948 | 0,931 [0,90; 0,96] |
| Hybrid med omrangering | alle lover og forskrifter | 0,927 | 0,913 | 0,892 [0,86; 0,92] |

Omrangering gir den største gevinsten. Strukturbiter som følger loven, er klart bedre enn tekstvinduer med fast lengde. Alle tall og sammenligninger står i [valideringsrapporten](docs/valideringsrapport.md).

Svarene er målt på alle 240 spørsmålene i testsettet. Riktighet er vurdert av en språkmodell mot referansesvarene, på de 198 spørsmålene som har svar.

| Generator og søk | Riktighet | Dekning kilder | Støtte | Riktige avslag | Feil avslag |
|---|--:|--:|--:|--:|--:|
| NorMistral 7B, BM25 | 0,631 [0,57; 0,69] | 0,056 | 0,405 | 0,000 | 0,010 |
| NorMistral 7B, hybrid med omrangering | 0,641 [0,58; 0,70] | 0,068 | 0,574 | 0,024 | 0,020 |
| Gemma 4 31B, BM25 | 0,821 [0,77; 0,87] | 0,768 | 0,988 | 0,976 | 0,157 |
| Gemma 4 31B, hybrid med omrangering | 0,917 [0,88; 0,95] | 0,869 | 0,984 | 0,976 | 0,051 |

Gemma med hybriden med omrangering gir best svar. Den viser oftest til riktig paragraf, kildene støtter påstandene, og den avstår når lovteksten ikke har svaret. Den lokale NorMistral skriver flytende norsk, men setter sjelden kilder i fast form og avstår nesten aldri.

Testsettet ble skrevet og gjennomgått av språkmodeller, og svarene er vurdert av en språkmodell.

## Kom i gang

Prosjektet krever Python 3.11 eller nyere.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,embed]"
```

Hvert steg er en kommando. Svarene fra språkmodellene lagres i en hurtigbuffer under `data/`, så stegene kan kjøres på nytt på samme maskin uten nye kall.

```bash
norsk-lov-rag corpus                      # last ned Lovdatas datasett og bygg tabellene
norsk-lov-rag references                  # finn henvisninger og bygg grafen
norsk-lov-rag index                       # lag tekstbiter og vektorer
norsk-lov-rag retrieve-eval --split test  # mål søket
norsk-lov-rag generate --split test       # lag svar
norsk-lov-rag judge --split test          # vurder svarene
norsk-lov-rag report --split test         # regn ut målene
norsk-lov-rag figures                     # lag figurene
norsk-lov-rag demo-export                 # lag filene til demoen
streamlit run app/Home.py                 # start demoen
```

Testsettet ligger ferdig i `testsett/`. Kommandoen `norsk-lov-rag testset` bygger det på nytt, og den trenger nøkler til de gratis API-ene.

Nøklene til de gratis API-ene leses fra filen `.env`, som ikke er med i repoet: `GROQ_API_KEY` for Groq og `GEMINI_API_KEY` for Gemini API. Den lokale modellen kjører i [Ollama](https://ollama.com). Se `configs/ollama/normistral.Modelfile` for hvordan den settes opp.

## Dokumentasjon

| Fil | Innhold |
|---|---|
| [metode.md](docs/metode.md) | Korpus, parsing, søk, svar og vurdering |
| [testsett.md](docs/testsett.md) | Hvordan testsettet ble laget, og tallene |
| [valideringsrapport.md](docs/valideringsrapport.md) | Planen for evalueringen og resultatene |
| [datakilder.md](docs/datakilder.md) | Data, modeller, tjenester og lisenser |
| [modellkort.md](docs/modellkort.md) | Formål, bruk, resultater og begrensninger |
| [personvern.md](docs/personvern.md) | Hva demoen gjør med spørsmålene |

## Data og lisens

Lovteksten kommer fra Lovdatas åpne datasett under NLOD 2.0. Kildeangivelsen er:

> Inneholder data under Norsk lisens for offentlige data (NLOD) tilgjengeliggjort av Stiftelsen Lovdata.

Se [DATA_NOTICE.md](DATA_NOTICE.md).

Koden er lisensiert under MIT. Se [LICENSE](LICENSE).

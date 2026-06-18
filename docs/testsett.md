# Testsettet

Testsettet måler hvor godt systemet finner de riktige bestemmelsene og svarer på spørsmål om skatterett. Det har 344 spørsmål om de ni dokumentene i skattekorpuset. Spørsmålene er skrevet, kontrollert og gjennomgått av språkmodeller, og ingen mennesker har lest dem før de ble tatt med.

## Typer spørsmål

| Type | Hva spørsmålet tester |
|---|---|
| Direkte | Et spørsmål med juridiske ord som kan besvares fra én bestemmelse. |
| Hverdagsspråk | Det samme behovet, spurt slik en vanlig person ville gjort det. |
| Krysshenvisning | Svaret krever både en bestemmelse og en bestemmelse den viser til. |
| Ubesvarbar | Korpuset har ikke svaret. Riktig svar er å si at teksten ikke svarer på spørsmålet. |

En del av spørsmålene er skrevet på nynorsk. Alle tekstene i skattekorpuset er på bokmål, så disse spørsmålene tester også om systemet tåler forskjellen mellom de to skriftformene.

## Utvalg

Bestemmelsene er trukket tilfeldig fra skattekorpuset med et fast frø. Antallet per dokument og per kapittel følger kvadratroten av hvor mange bestemmelser de har. Da tar ikke de største lovene hele settet. Hver paragraf er brukt høyst én gang. Bare gjeldende paragrafer er med, og overgangsregler og svært korte eller svært lange ledd er holdt utenfor.

Til krysshenvisningene er parene hentet fra referansegrafen. Bare henvisninger som er funnet direkte i teksten er brukt: henvisninger i samme dokument, henvisninger som nevner loven ved navn, og henvisninger med lovens dato og nummer.

De ubesvarbare spørsmålene er skrevet for seg. De gjelder satser og beløpsgrenser som Stortinget fastsetter hvert år, praktiske spørsmål til Skatteetaten, skatteregler i andre land og spørsmål utenfor skatteretten. Stortingets skattevedtak er ikke med i korpuset.

## Skriving

Spørsmålene er skrevet av `openai/gpt-oss-120b` gjennom gratisnivået hos Groq. For hver bestemmelse skrev modellen et spørsmål, et kort referansesvar og hvilke ledd svaret bygger på. Modellen hører til en annen modellfamilie enn modellene som skal svare på spørsmålene senere i prosjektet. Alle kall er lagret, så settet kan bygges på nytt fra de samme svarene.

## Filtre og kontroll

Et spørsmål ble fjernet hvis det

- viste til tekst som leseren ikke ser, for eksempel «første ledd», et paragrafnummer eller «tekst A»;
- var skrevet i feil skriftform;
- tok mer enn fem ord på rad fra bestemmelsen, slik at ordsøk ikke får en urettferdig fordel;
- lignet for mye på et annet spørsmål, målt som overlapp mellom grupper på fire tegn.

Deretter kontrollerte `qwen/qwen3.8-27b` (også hos Groq) at hvert spørsmål kan besvares fra de valgte leddene alene, og at referansesvaret stemmer med teksten. For krysshenvisninger kontrollerte modellen også at begge bestemmelsene trengs.

De ubesvarbare spørsmålene ble kontrollert med søk. Et BM25-søk med norsk ordstamming fant de åtte nærmeste tekstene, og modellen sjekket at ingen av dem svarer på spørsmålet. Spørsmål om satser, om Skatteetaten og om andre land ble søkt i både skattekorpuset og hele korpuset, fordi hele korpuset har noen forskrifter med beløp. Spørsmål utenfor skatteretten ble søkt i skattekorpuset.

## Gjennomgang

Til slutt gikk `gemini-3.6-flash` gjennom hvert spørsmål, med fast versjon og gjennom gratisnivået hos Google. Den er en sterkere modell enn modellen som skrev spørsmålene. Den fikk hele teksten til bestemmelsene, og for ubesvarbare spørsmål de fem beste søketreffene. Den merket hvert spørsmål som korrekt, feil eller uklart, med en kort begrunnelse. Bare spørsmål merket korrekt er med. Uklare spørsmål er fjernet, ikke rettet. Krysshenvisninger ble sendt i grupper på ti spørsmål og de andre typene i grupper på tjue.

Gratisnivået gir få kall til denne modellen. 34 spørsmål som hadde passert filtrene og kontrollen, ble derfor ikke gjennomgått. De er ikke med i settet.

## Tall

Tabellen viser hvor mange spørsmål som ble skrevet, hvor mange som ble fjernet i hvert steg, og hvor mange som er med til slutt.

| Type | Skrift | Skrevet | Filtre | Kontroll | Gjennomgang | Ikke gjennomgått | Med | Utvikling | Test |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Direkte | nb | 130 | 44 | 1 | 6 | 5 | 74 | 22 | 52 |
| Direkte | nn | 30 | 5 | 0 | 0 | 5 | 20 | 6 | 14 |
| Hverdagsspråk | nb | 130 | 1 | 2 | 11 | 7 | 109 | 33 | 76 |
| Hverdagsspråk | nn | 30 | 2 | 1 | 1 | 7 | 19 | 6 | 13 |
| Krysshenvisning | nb | 150 | 23 | 37 | 19 | 10 | 61 | 18 | 43 |
| Ubesvarbar | nb | 55 | 0 | 3 | 0 | 0 | 52 | 16 | 36 |
| Ubesvarbar | nn | 10 | 0 | 0 | 1 | 0 | 9 | 3 | 6 |
| Sum | | 535 | 75 | 44 | 38 | 34 | 344 | 104 | 240 |

De fleste direkte spørsmålene som ble fjernet, tok for mange ord fra bestemmelsen. Krysshenvisningene mistet flest spørsmål i kontrollen og i gjennomgangen. Av de 80 krysshenvisningene som ble gjennomgått, ble 19 fjernet. For direkte spørsmål og hverdagsspråk var andelen mellom 6 og 9 prosent. Krysshenvisningene ble sendt i mindre grupper, så vi kan ikke skille virkningen av gruppestørrelsen fra virkningen av typen.

## Deling

Hver kombinasjon av type og skriftform er delt for seg. 30 prosent går til utviklingssettet og resten til testsettet. Rekkefølgen bestemmes av en hash av spørsmålets id, så delingen er fast.

Utviklingssettet har 104 spørsmål og brukes til å justere systemene. Testsettet har 240 spørsmål. Det er frosset og brukes bare til de endelige resultatene. Begge ligger i mappen `testsett/`. Filen `manifest.json` der har SHA-256 for hver fil, korpusets hash, modellene og versjonene av instruksene.

## Kjente svakheter

Ingen mennesker har lest spørsmålene, og gjennomgangsmodellen kan ta feil. Ett eksempel er `cross_reference-nb-009`, som ble godkjent. Svaret bruker grensen på 3 millioner kroner i merverdiavgiftsloven § 2-1 annet ledd, men den grensen gjelder bare salg av billetter til idrettsarrangementer. Spørsmålet er beholdt fordi settet følger metoden uten håndrettinger.

Referansesvarene bygger bare på de valgte leddene. Andre regler kan også ha betydning for svaret. Nynorsk-spørsmålene har noen bokmålsord. Fordelingen er heller ikke jevn: det er færre krysshenvisninger og nynorsk-spørsmål enn de andre typene.

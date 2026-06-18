# Personvern i demoen

Denne filen forklarer hva som skjer med spørsmålene som skrives inn i demoen.

## Ikke skriv personopplysninger

Demoen ber brukeren om ikke å skrive navn, fødselsnummer, adresser eller andre opplysninger om seg selv eller andre. Et spørsmål om lovteksten trenger ikke slike opplysninger. Skriv spørsmålet generelt, for eksempel «Kan man trekke fra reiseutgifter mellom hjem og arbeid?».

## Hva som sendes ut

Søket i lovteksten skjer på serveren der demoen kjører. Når brukeren ber om et svar, sendes spørsmålet og utdragene fra lovteksten til Google Gemini API, som lager svaret med modellen `gemma-4-31b-it`. Demoen sier dette ved siden av feltet der spørsmålet skrives.

Demoen bruker gratis nivå av Gemini API. For brukere i EØS gjelder vilkårene for betalt bruk også på gratis nivå. Det betyr at Google ikke bruker innholdet til å forbedre sine produkter. Vilkårene er beskrevet nærmere i `datakilder.md`.

## Hva som lagres

Teksten i spørsmålet lagres ikke. Loggen har bare en hash av spørsmålet, lengden på det, hvor lang tid svaret tok, antall tokens og utfallet, for eksempel svar, avslag eller at grensen er nådd. Hashen gjør det mulig å se om samme spørsmål kommer igjen, uten at teksten kan leses. IP-adressen brukes bare til å telle forespørsler per dag og skrives ikke i loggen.

## Grenser

Hver IP-adresse kan få fem svar fra modellen per dag, og hele demoen kan få 100 svar per dag. Et spørsmål kan ha høyst 500 tegn. Tellerne starter på null hver dag ved midnatt UTC. Når grensen for dagen er nådd, eller den gratis kvoten hos Google er brukt opp, viser demoen ferdige svar på eksempelspørsmål i stedet. Disse svarene er laget på forhånd, og da sendes ingenting til Google.

Sidene «Eksempler» og «Sammenlign søk» kaller aldri en språkmodell. Et spørsmål som skrives på «Sammenlign søk», forlater ikke serveren.

## Nøkler

Nøkkelen til Gemini API ligger i innstillingene til tjenesten der demoen kjører, og aldri i koden eller i repoet.

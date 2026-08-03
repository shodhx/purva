# PURVA Annotation Guidelines

This document is the human annotation manual for the PURVA Bhojpuri sentiment
dataset. It implements PROTOCOL.md §3 (label scheme) and restates the relevant
blinding rules from §9. Annotators should read this document in full before
labeling any sentence, and should re-read section (c) whenever a sentence
feels ambiguous.

## (a) Task overview and the two-stage scheme

Each sentence in the corpus is labeled in two stages.

**Stage A — subjectivity: `{objective, subjective}`**

- `objective` — the sentence reports a fact, event, description, or piece of
  information without the author expressing a personal evaluative stance
  toward it. News reporting, encyclopedic content, and plain narration of
  events are typically objective, even when the vocabulary involved is vivid
  or the event itself is emotionally weighty (see case 5).
- `subjective` — the sentence expresses (or clearly implies) an evaluative,
  affective, or emotional stance — the author's or a quoted speaker's opinion,
  feeling, praise, complaint, judgment, or reaction. Verse, commentary,
  devotional writing, and first-person narration are frequently subjective.

**Stage B — polarity, applied only to sentences labeled `subjective`:
`{positive, negative, neutral, mixed}`**

- `positive` — the expressed stance is favorable, appreciative, hopeful, or
  celebratory.
- `negative` — the expressed stance is critical, sorrowful, angry, fearful,
  or disapproving.
- `neutral` — the sentence is subjective (it expresses a stance, not a bare
  fact) but the stance is mild, hedged, ambivalent-but-flat, or otherwise
  carries no clear positive or negative charge.
- `mixed` — the sentence clearly expresses **both** a positive and a negative
  component, and neither dominates the other.

**Final label space:** `{objective, positive, negative, neutral, mixed}` — a
sentence labeled `objective` in Stage A never receives a Stage B label; every
`subjective` sentence receives exactly one of the four Stage B labels.

**Additional per-sentence outputs** (recorded alongside the stage labels, for
both human annotators and the judge committee):

- `normalized_domain` — a short (one-to-three word) topic label, e.g.
  "politics", "festival", "family", "cinema", "farming".
- `narrative_voice` — `first_person`, `third_person`, or `mixed`.
- `sentiment_target` — nullable free text naming who/what the sentiment (if
  any) is directed at, e.g. "the government", "the narrator's mother". Null
  for `objective` sentences and for `subjective` sentences with no clear
  target.
- `confidence` — a 0–1 self-reported confidence in the label.
- `rationale` — one sentence, maximum, explaining the label.

## (b) Decision procedure

Follow these steps in order for every sentence:

1. **Read the whole sentence once, plainly**, without pattern-matching on
   individual "positive" or "negative" words in isolation.
2. **Stage A first:** does the sentence assert a fact/event/description with
   no evaluative stance from the author, or does it express or imply a
   stance? Decide `objective` vs `subjective` before thinking about polarity
   at all.
3. **If objective, stop** — record `objective`, leave polarity fields null,
   fill in `normalized_domain`, `narrative_voice`, and `rationale`.
4. **If subjective, check negation scope and sarcasm/irony markers before
   reading off surface polarity words** — a positive-looking word under
   negation or under sarcastic framing flips the actual stance (see cases 3
   and 9).
5. **Determine dominant polarity:** if exactly one clear valence (favorable or
   unfavorable) dominates, label `positive` or `negative`. If two clearly
   opposed valences are both present and neither dominates, label `mixed`. If
   the stance is present but weak, hedged, or flat, label `neutral`.
6. **Identify `narrative_voice`** from whose perspective the sentence is
   told.
7. **Identify `sentiment_target`** if the subjective stance is clearly
   directed at a person, group, institution, or thing; otherwise leave null.
8. **Fill `normalized_domain`** with the sentence's topic in one to three
   words.
9. **Set `confidence` honestly.** A genuinely torn call should get a lower
   confidence score rather than being forced into false certainty.
10. **Write `rationale`** in one sentence, stating the specific cue (word,
    structure, or context) that drove the label.

## (c) Twelve worked hard cases

Each case below gives a realistic Bhojpuri example, the correct label, and a
one-line justification. These are the cases annotators most often get wrong;
when in doubt, match the current sentence against the closest case here.

**1. Verse metaphor — label the expressed sentiment, not the literal image**

> मन के बगिया में उदासी के कँटा गड़ल बा।
> ("Thorns of sadness have pierced the garden of my heart.")

Label: `subjective`, `negative`. *Justification:* the garden/thorn imagery is
figurative; the sentiment being expressed through the metaphor is sorrow, so
the metaphor is decoded to its emotional content, not left as an "objective"
description of a garden.

**2. Quoted Hindi/other-language speech inside a Bhojpuri frame**

> ऊ हिंदी में कहलस, "मुझे माफ़ कर दो," आ फेर रोवे लागल।
> (He said in Hindi, "Forgive me," and then started crying.)

Label: `subjective`, `negative`. *Justification:* the language of the quoted
speech (Hindi) is irrelevant to labeling — label the sentiment expressed by
the sentence as a whole (regret, distress), not the script/language of an
embedded quotation.

**3. Sarcasm / satire (बतकूचन-style)**

> वाह रे सरकार, बिजली त अइसन दिहलू कि दिनहूँ में अन्हार बा।
> ("Wow, what a government — you gave us electricity so 'good' that it's dark
> even in daytime.")

Label: `subjective`, `negative`. *Justification:* surface praise words ("वाह",
"अइसन दिहलू") are sarcastic; the actual stance is criticism of poor electricity
supply. Satirical बतकूचन-style commentary should be read for its real target,
not its surface lexicon.

**4. Mixed sentiment in one sentence**

> बेटा पास त हो गइल बाकिर नंबर बहुते कम आइल, खुशी असो गम दुनु बा।
> ("The son passed, but with very low marks — there's both happiness and
> sorrow.")

Label: `subjective`, `mixed`. *Justification:* both a positive component
(passing) and a negative component (low marks) are explicitly present and
neither is subordinated to the other.

**5. Objective news statement containing emotive vocabulary**

> दुर्घटना में तीन गो मजदूर के दर्दनाक मौत हो गइल।
> ("Three laborers died a painful death in the accident.")

Label: `objective`. *Justification:* this is fact reporting in news register;
"दर्दनाक" ("painful/tragic") is a conventional descriptor for this kind of
event in news prose, not the author inserting a personal evaluative opinion.
Test: does the sentence do anything beyond stating what happened? Here, no.

**6. Devotional praise (subjective positive, not objective)**

> हे राम, रउरा त दया के सागर बानी।
> ("O Ram, you are an ocean of compassion.")

Label: `subjective`, `positive`. *Justification:* devotional address and
praise is an expression of admiration/reverence, even though it is phrased as
a description ("you are an ocean of compassion") — descriptions of a deity's
virtues in a devotional register are evaluative praise, not neutral fact.

**7. Proverb / aphorism**

> जे गरजेला ऊ बरसेला ना।
> ("Those who thunder don't rain" — i.e. all talk, no action.)

Label (as a standalone, context-free proverb): `objective`. *Justification:*
a proverb stated in isolation, with no addressed target, is a generalized
statement of received wisdom, not an evaluative stance about a specific
person or event. If the same proverb is used pointedly to criticize a named
person or group in context, re-label `subjective`/`negative` for that
instance — the proverb's *use*, not its wording, determines the label.

**8. Rhetorical question**

> का हमरा ई सब झेलल जरूरी रहे?
> ("Was it really necessary for me to endure all this?")

Label: `subjective`, `negative`. *Justification:* rhetorical questions are
not genuine information requests; this one voices complaint/frustration, so
it is labeled for the sentiment it conveys, not treated as a neutral
question.

**9. Negation flips**

> ई फिलिम नीक नइखे।
> ("This film is not good.")

Label: `subjective`, `negative`. *Justification:* the presence of the
positive word "नीक" ("good") is irrelevant once negation ("नइखे") is applied —
read the full scope of negation before assigning polarity from an isolated
keyword.

**10. Headline fragments**

> भोजपुरी सम्मेलन: राज्य न रोटी।
> ("Bhojpuri Conference: No State, No Bread.")

Label: `objective`. *Justification:* verbless headline fragments that name a
topic/event default to `objective` — there is no stated author evaluation,
only a compressed topical label, even when the fragment alludes to grievance.
Only label such a fragment `subjective` if the noun phrase itself is
unambiguously an evaluative claim rather than a topic tag.

**11. First-person lament verse**

> हम अकेले सहीं ई गम, केहू ना बुझेला मोर पीर।
> ("I alone bear this sorrow, no one understands my pain.")

Label: `subjective`, `negative`. *Justification:* explicit first-person
expression of sorrow and isolation; unambiguous negative affect from the
speaker's own voice.

**12. Neutral opinion — hedging ("ठीक बा")**

> ई योजना ठीक बा, बाकिर समय लागी।
> ("This scheme is okay, but it will take time.")

Label: `subjective`, `neutral`. *Justification:* "ठीक बा" ("it's fine/okay") is
a hedged, lukewarm evaluation — it is a stance (so not `objective`), but it
carries no clear positive or negative charge, so it is `neutral` rather than
`positive`.

## (d) Language-validation task

In addition to sentiment labeling, annotators fill a `human_language` column
for a 500-row sample drawn for the LID (language identification) validation
described in PROTOCOL.md §2. This task is done **blind to the model's
`lid_verdict`/`lid_model_label`** — annotators must not see the model's
prediction while making this judgment.

`human_language ∈ {bhojpuri, hindi, maithili, other}`. Decision rule:

- **`hindi`** — the sentence is fully grammatical standard Hindi with no
  Bhojpuri morphology, lexis, or verb-agreement pattern present anywhere in
  the sentence.
- **`bhojpuri`** — any Bhojpuri-specific morphology or lexis is present
  (e.g. verb forms in `-ल/-ली/-लस/-लीं`, copulas `बा/बाड़ें/बानी`, pronouns like
  `हमरा/रउआ/ऊ`, or characteristic Bhojpuri vocabulary), even if the sentence is
  mixed with Hindi words or code-switches mid-sentence. Bhojpuri features
  present anywhere in the sentence outweigh Hindi features present elsewhere
  in the same sentence.
- **`maithili`** — the sentence is identifiable as Maithili rather than
  Bhojpuri or Hindi (distinct Maithili morphology/lexis, no Bhojpuri
  features).
- **`other`** — none of the above (a different language entirely, or text too
  degenerate/fragmentary to judge).
- **Genuinely undecidable** — if, after applying the above rules, the
  annotator still cannot confidently decide, they do **not** guess. They mark
  the row in the separate `flag` column (e.g. `undecidable`) and may leave
  `human_language` blank; such rows are reviewed by adjudication rather than
  scored as a plain error.

## (e) Blinding rules (restated from PROTOCOL.md §9)

- Human annotators never see model labels, model confidence scores, or model
  rationales prior to adjudication — this applies to both the sentiment task
  and the language-validation task above.
- Test-split items are never used for model selection or threshold
  selection.
- Every deviation from this protocol is logged with its date in the
  CHANGELOG section of PROTOCOL.md.

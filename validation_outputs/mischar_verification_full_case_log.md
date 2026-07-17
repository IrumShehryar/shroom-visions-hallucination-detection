# Full Case-by-Case Log: Mischaracterization Verification Test

Total cases: 51 (across dev-pool and genuinely-fresh held-out batches)

| # | Batch | Known GT | Sample ID | Flagged span | Haiku label/conf | Sonnet verdict/conf |
|---|---|---|---|---|---|---|
| 1 | dev | known-FP | train-en-2657 | There are no words in green text in the image. | mischaracterization / 0.9 | reject / 0.95 |
| 2 | dev | known-FP | train-en-2037 | not composed of circular tubes | mischaracterization / 0.85 | reject / 0.75 |
| 3 | dev | known-FP | train-en-1766 | white stripes on its back | mischaracterization / 0.85 | confirm / 0.85 |
| 4 | dev | known-FP | train-en-11473 | The wings and upper body are a mix of dark brown and grayish... | mischaracterization / 0.85 | reject / 0.55 |
| 5 | dev | known-FP | train-en-11409 | paraglider wing | mischaracterization / 0.85 | confirm / 0.85 |
| 6 | dev | known-FP | train-en-992 | green flooring | mischaracterization / 0.85 | reject / 0.7 |
| 7 | dev | known-FP | train-en-1693 | bralette | mischaracterization / 0.85 | confirm / 0.85 |
| 8 | dev | known-FP | train-en-1478 | gold ring | mischaracterization / 0.85 | confirm / 0.9 |
| 9 | dev | known-FP | train-en-11050 | Pumpkins are usually green or orange | mischaracterization / 0.7 | confirm / 0.75 |
| 10 | dev | known-TP | train-en-11357 | It extends from the shoulder to the elbow area, covering the... | mischaracterization / 0.95 | confirm / 0.98 |
| 11 | dev | known-TP | train-en-11736 | circular base that is not visible from the current angle | mischaracterization / 0.85 | confirm / 0.9 |
| 12 | dev | known-TP | train-en-920 | long barrel | mischaracterization / 0.85 | confirm / 0.7 |
| 13 | dev | known-TP | train-en-1959 | multiple layers of thick brown leather | mischaracterization / 0.85 | confirm / 0.7 |
| 14 | dev | known-TP | train-en-11886 | primarily black | mischaracterization / 0.85 | confirm / 0.85 |
| 15 | dev | known-TP | train-en-966 | rectangular blade | mischaracterization / 0.95 | confirm / 0.9 |
| 16 | dev | known-TP | train-en-11402 | This is not an orange, but rather a grapefruit. | mischaracterization / 0.85 | PARSE_ERROR / None |
| 17 | dev | known-TP | train-en-11218 | two small front wheels connected by a frame bar | mischaracterization / 0.85 | confirm / 0.97 |
| 18 | dev | known-FP | train-en-11837 | cucumber | mischaracterization / 0.95 | reject / 0.75 |
| 19 | dev | known-FP | train-en-1499 | wide belt | mischaracterization / 0.95 | reject / 0.9 |
| 20 | dev | known-FP | train-en-11078 | heart-like shape | mischaracterization / 0.85 | reject / 0.6 |
| 21 | dev | known-TP | train-en-11535 | Yes, the corn on the cob in the image is yellow in color | mischaracterization / 0.95 | confirm / 0.97 |
| 22 | dev | known-TP | train-en-11171 | Bactrian camels | mischaracterization / 0.95 | confirm / 0.98 |
| 23 | dev | known-TP | train-en-2083 | This wasp does not appear to have visible wings | mischaracterization / 0.95 | confirm / 0.75 |
| 24 | dev | known-TP | train-en-11198 | single-seat | mischaracterization / 0.95 | confirm / 0.97 |
| 25 | dev | known-TP | train-en-10909 | The mug in the image appears to have a handle | mischaracterization / 0.95 | confirm / 0.97 |
| 26 | dev | known-TP | train-en-10221 | blue banner | mischaracterization / 0.85 | confirm / 0.9 |
| 27 | dev | known-TP | train-en-1542 | push-pull | mischaracterization / 0.85 | confirm / 0.9 |
| 28 | dev | known-TP | train-en-9663 | on the right side of the image | mischaracterization / 0.85 | confirm / 0.8 |
| 29 | dev | known-TP | train-en-1836 | from about mid-neck down to just above the knees | mischaracterization / 0.75 | reject / 0.7 |
| 30 | dev | known-TP | train-en-1339 | handbag, which is tucked in under her arm | mischaracterization / 0.85 | confirm / 0.9 |
| 31 | dev | known-TP | train-en-1767 | red beak | mischaracterization / 0.95 | confirm / 0.85 |
| 32 | dev | known-TP | train-en-683 | white or pale yellow bands | mischaracterization / 0.85 | reject / 0.75 |
| 33 | dev | known-TP | train-en-1517 | no red car visible | mischaracterization / 0.95 | confirm / 0.98 |
| 34 | dev | known-TP | train-en-11398 | pinkish | mischaracterization / 0.85 | confirm / 0.9 |
| 35 | dev | known-TP | train-en-11309 | cucumbers | mischaracterization / 0.95 | reject / 0.75 |
| 36 | dev | known-TP | train-en-1538 | right hand | mischaracterization / 0.85 | PARSE_ERROR / None |
| 37 | dev | known-TP | train-en-1274 | light-colored stripe running across the forehead | mischaracterization / 0.85 | confirm / 0.72 |
| 38 | dev | known-TP | train-en-995 | plastic | mischaracterization / 0.75 | reject / 0.75 |
| 39 | held-out | known-FP | train-en-477 | lighter, almost white or lavender stripes | mischaracterization / 0.9 | reject / 0.85 |
| 40 | held-out | known-FP | train-en-2371 | cracked asphalt or concrete | mischaracterization / 0.85 | reject / 0.6 |
| 41 | held-out | known-FP | train-en-2057 | propellers | mischaracterization / 0.9 | reject / 0.75 |
| 42 | held-out | known-FP | train-en-1828 | The yellow and teal containers hold whole coconut fruits tha... | mischaracterization / 0.85 | confirm / 0.85 |
| 43 | held-out | known-FP | train-en-11101 | blue SUV | mischaracterization / 0.85 | reject / 0.75 |
| 44 | held-out | known-FP | train-en-830 | bright | mischaracterization / 0.75 | reject / 0.6 |
| 45 | held-out | known-FP | train-en-469 | does not appear to have stitching | mischaracterization / 0.85 | reject / 0.93 |
| 46 | held-out | known-TP | train-en-876 | still has its skin on | mischaracterization / 0.95 | confirm / 0.97 |
| 47 | held-out | known-TP | train-en-1483 | rectangular blade | mischaracterization / 0.9 | confirm / 0.9 |
| 48 | held-out | known-TP | train-en-11139 | rose or a floral pattern | mischaracterization / 0.85 | reject / 0.75 |
| 49 | held-out | known-TP | train-en-2207 | grayish areas around its neck | mischaracterization / 0.85 | confirm / 0.8 |
| 50 | held-out | known-TP | train-en-1727 | suspended from the ceiling by chains or straps | mischaracterization / 0.9 | confirm / 0.85 |
| 51 | held-out | known-TP | train-en-9671 | blue and red text | mischaracterization / 0.85 | reject / 0.97 |

## Full detail (span text, Haiku reason, Sonnet reason) per case

### 1. train-en-2657 (dev, known-FP)
- **Flagged span:** 'There are no words in green text in the image.'
- **Haiku:** label=mischaracterization, confidence=0.9
  - reason: The image clearly shows red text boxes on the right side, not green text; the response incorrectly characterizes the text color as absent rather than identifying the actual red text visible.
- **Sonnet:** verdict=reject, confidence=0.95
  - reason: The claim only addresses green text, which is indeed absent; the presence of red text elsewhere does not make the statement about green text false.

### 2. train-en-2037 (dev, known-FP)
- **Flagged span:** 'not composed of circular tubes'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The fire hose visible in the image is clearly made of circular/tubular material with a round cross-section, which is the standard construction for fire hoses; the response incorrectly claims it is not composed of circular tubes.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: Flat-lay fire hoses like the one shown are typically flattened (not round in cross-section) when empty and coiled, so the response's claim is a reasonable description rather than a clear mischaracterization.

### 3. train-en-1766 (dev, known-FP)
- **Flagged span:** 'white stripes on its back'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The skunk has white stripes on its body and tail, but the pattern is not simply stripes on the back—it has a more complex spotted/blotched white pattern on the head and body, not organized as linear back stripes.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The animal shown is a spotted skunk with a mottled/spotted pattern of white blotches on the head and broken white bands on the body, not simple linear stripes on the back.

### 4. train-en-11473 (dev, known-FP)
- **Flagged span:** 'The wings and upper body are a mix of dark brown and grayish-brown'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The upper body and wings appear predominantly dark gray or charcoal, not brown or grayish-brown as claimed.
- **Sonnet:** verdict=reject, confidence=0.55
  - reason: The wings and upper body show dark brown tones consistent with the response's description, not purely gray or charcoal as the flag claims.

### 5. train-en-11409 (dev, known-FP)
- **Flagged span:** 'paraglider wing'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The image shows a parafoil/power kite design (as indicated by the filename and visible branding), not a paraglider wing which is used for paragliding sports with a pilot harness.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The image shows a foil kite (branded 'QUEST' with kite-strength diamond symbols) flown on lines, not a paraglider wing which is a much larger canopy used with a pilot harness for flight.

### 6. train-en-992 (dev, known-FP)
- **Flagged span:** 'green flooring'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The flooring visible in the image is gray/concrete, not green. The green area appears to be artificial turf or a green surface to the left side, not the main flooring of the space.
- **Sonnet:** verdict=reject, confidence=0.7
  - reason: There is indeed a visible green surface (artificial turf) in the lower left of the image, so describing 'green flooring' in the scene is reasonably accurate even though it's not the entire floor.

### 7. train-en-1693 (dev, known-FP)
- **Flagged span:** 'bralette'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The garment shown is a strapless bra with underwire and structured cups, not a bralette, which is typically an unstructured, wire-free style.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The garment shown is a structured, molded-cup strapless bra with underwire, not a soft unstructured bralette.

### 8. train-en-1478 (dev, known-FP)
- **Flagged span:** 'gold ring'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The girl is wearing a bracelet or band on her wrist, not a ring, and it appears to be dark/gray rather than gold.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The visible accessory is a dark/silver bracelet on her wrist, not a gold ring, so the response's mention of a gold ring is inaccurate.

### 9. train-en-11050 (dev, known-FP)
- **Flagged span:** 'Pumpkins are usually green or orange'
- **Haiku:** label=mischaracterization, confidence=0.7
  - reason: The response claims pumpkins are 'usually green or orange' but orange is by far the most typical color for common pumpkin varieties; green pumpkins are less common and not a standard description of what pumpkins 'usually' are.
- **Sonnet:** verdict=confirm, confidence=0.75
  - reason: Pumpkins are predominantly known as orange, so stating they are 'usually green or orange' overstates green as a typical color, which is a minor mischaracterization.

### 10. train-en-11357 (dev, known-TP)
- **Flagged span:** 'It extends from the shoulder to the elbow area, covering the upper arm'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The sweater vest is sleeveless with a V-neck; it does not extend to the elbow or cover the upper arm—the long-sleeved shirt underneath is visible on both arms.
- **Sonnet:** verdict=confirm, confidence=0.98
  - reason: The garment is a sleeveless sweater vest, with the blue shirt's sleeves visible covering the arms, so it does not extend to the elbow or cover the upper arm.

### 11. train-en-11736 (dev, known-TP)
- **Flagged span:** 'circular base that is not visible from the current angle'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The base is clearly visible and is not circular but rather an oval/elongated footprint, and it is fully visible in the image.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The base is clearly visible in the image as an oval/rounded stand shape, contradicting the claim that it is 'not visible from the current angle'.

### 12. train-en-920 (dev, known-TP)
- **Flagged span:** 'long barrel'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: This is a bullpup rifle design where the barrel is actually relatively short compared to traditional rifle configurations; the bullpup platform places the action near the stock, resulting in a compact overall length despite the barrel length being moderate.
- **Sonnet:** verdict=confirm, confidence=0.7
  - reason: This is a Kel-Tec RDB bullpup rifle where the compact overall length comes from the action being positioned behind the trigger, meaning the barrel itself is not unusually long despite the rifle's total length.

### 13. train-en-1959 (dev, known-TP)
- **Flagged span:** 'multiple layers of thick brown leather'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The belt is a single piece of leather wrapped around, not multiple stacked layers; the structure shows one continuous leather piece with belt straps attached.
- **Sonnet:** verdict=confirm, confidence=0.7
  - reason: The wide belt/cinch appears to be a single continuous piece of leather shaped into a wide waist band with two separate buckle straps attached, not multiple stacked layers of leather.

### 14. train-en-11886 (dev, known-TP)
- **Flagged span:** 'primarily black'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The heads are not primarily black but rather olive-green and yellowish-green, which are distinctive features visible in the image.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The penguins' heads are predominantly pale yellow-green on top with a brownish-black band around the eyes/cheeks, not primarily black.

### 15. train-en-966 (dev, known-TP)
- **Flagged span:** 'rectangular blade'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The visible grater drum has a cylindrical shape with holes arranged in patterns, not a rectangular blade; the rectangular attachments shown are separate shredding cylinders, not the main grater component.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The grater drums in the image are cylindrical with round or slotted holes, not rectangular blades, so the response's description is inaccurate.

### 16. train-en-11402 (dev, known-TP)
- **Flagged span:** 'This is not an orange, but rather a grapefruit.'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The image shows a grapefruit that has been cut in half, displaying pink/red flesh interior, which is consistent with grapefruit; however, the prompt asks about an orange and the filename suggests it may be labeled as such, making the definitive statement that 'this is not an orange' potentially incorrect without additional context.
- **Sonnet:** verdict=PARSE_ERROR, confidence=None
  - reason: {"verdict": "reject", "confidence": 0.75, "reason": "The image shows fruit with orange-colored rind (peel) but pink/red flesh inside, which visually resembles a Cara Cara or blood orange rather than a

### 17. train-en-11218 (dev, known-TP)
- **Flagged span:** 'two small front wheels connected by a frame bar'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The image shows a single front wheel, not two small front wheels; the frame bar is the main bicycle frame, not a connection between two front wheels.
- **Sonnet:** verdict=confirm, confidence=0.97
  - reason: The bicycle has a single front wheel (with the second wheel visible being the rear wheel from an odd angle), and the frame bar is simply the bike's main frame, not a connector between two front wheels.

### 18. train-en-11837 (dev, known-FP)
- **Flagged span:** 'cucumber'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The object in the image is a lemon (yellow, round fruit with visible seeds/texture), not a cucumber, as confirmed by the filename and visual characteristics.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The image shows a yellow, oval fruit with spiky trichomes and a green stem/calyx attachment, consistent with a small yellow tomato variety rather than a lemon, and the response never definitively claimed it was a lemon or cucumber but was reasoning about cucumber shape in the context of the question.

### 19. train-en-1499 (dev, known-FP)
- **Flagged span:** 'wide belt'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The prompt asks if the belt is thin, and the response claims it is a 'wide belt,' but the belt visible in the image is a narrow black band, not wide.
- **Sonnet:** verdict=reject, confidence=0.9
  - reason: The black obi visible in the image is a wide sash band spanning much of the torso, not a narrow belt, so calling it a 'wide belt' is accurate.

### 20. train-en-11078 (dev, known-FP)
- **Flagged span:** 'heart-like shape'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The fruit is clearly ovate/elliptical, not heart-shaped; it lacks the distinctive pointed bottom characteristic of a heart shape.
- **Sonnet:** verdict=reject, confidence=0.6
  - reason: The fruit does show a slightly tapered, indented top with a broader rounded bottom, giving it a plausible heart-like appearance, so the description is a reasonable subjective characterization rather than clearly wrong.

### 21. train-en-11535 (dev, known-TP)
- **Flagged span:** 'Yes, the corn on the cob in the image is yellow in color'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The corn in the image is predominantly red/burgundy in color, not yellow. While there is a small amount of pale/cream coloring visible on the kernels, the dominant color is clearly red/purple, not yellow.
- **Sonnet:** verdict=confirm, confidence=0.97
  - reason: The corn kernels in the image are predominantly deep red/burgundy, not yellow, with only the husks being green/pale colored.

### 22. train-en-11171 (dev, known-TP)
- **Flagged span:** 'Bactrian camels'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The response incorrectly identifies the camel as a Bactrian camel, which have two humps. This camel has one hump, making it a Dromedary (Arabian) camel, as the filename and single-hump anatomy confirm.
- **Sonnet:** verdict=confirm, confidence=0.98
  - reason: The camel in the image clearly has a single hump, making it a dromedary, not a Bactrian camel with two humps.

### 23. train-en-2083 (dev, known-TP)
- **Flagged span:** 'This wasp does not appear to have visible wings'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The image shows a velvet ant (Mutillidae), which is not a wasp but a wingless wasp family member; however, more critically, velvet ants are known to be wingless or have greatly reduced wings, making the phrasing misleading as it suggests wings might be present but hidden.
- **Sonnet:** verdict=confirm, confidence=0.75
  - reason: The image shows a velvet ant (Mutillidae), a wingless wasp family, so calling it a 'wasp' with possibly hidden wings is misleading despite the observation about no visible wings being technically accurate.

### 24. train-en-11198 (dev, known-TP)
- **Flagged span:** 'single-seat'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The aircraft visibly has two seats (dual seating arrangement for pilot and co-pilot/cadet), not a single seat.
- **Sonnet:** verdict=confirm, confidence=0.97
  - reason: The image clearly shows two seats side by side in the cockpit, contradicting the claim of a single-seat aircraft.

### 25. train-en-10909 (dev, known-TP)
- **Flagged span:** 'The mug in the image appears to have a handle'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The bowls shown in the image do not have handles; they are handleless ceramic bowls with decorative floral patterns.
- **Sonnet:** verdict=confirm, confidence=0.97
  - reason: The image shows handleless ceramic tea bowls (chawan) with floral patterns, not mugs with handles.

### 26. train-en-10221 (dev, known-TP)
- **Flagged span:** 'blue banner'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The banner on the cross appears to be yellow/gold, not blue.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The cross carried in the image has a red/orange banner with text, not a blue one.

### 27. train-en-1542 (dev, known-TP)
- **Flagged span:** 'push-pull'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: This is a light rail vehicle (LRV) or articulated tram, not a push-pull train configuration. Push-pull refers to a specific locomotive-hauled configuration, which does not apply to this self-propelled multiple-unit vehicle.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The image shows a self-propelled light rail vehicle/tram (Hong Kong LRT) with pantograph power collection, not a locomotive-hauled push-pull train configuration.

### 28. train-en-9663 (dev, known-TP)
- **Flagged span:** 'on the right side of the image'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The 'SALE' signs are visible on the right side of the image, but they do not show a '75%' discount reduction. The visible 'SALE' signs on the right appear to be generic sale signage without specific percentage information visible.
- **Sonnet:** verdict=confirm, confidence=0.8
  - reason: The 75% discount sign is visible in the pink window on the left-center area (near Silver Nail/XTC shops), while the right side only shows generic 'SALE' signs without visible percentage figures.

### 29. train-en-1836 (dev, known-TP)
- **Flagged span:** 'from about mid-neck down to just above the knees'
- **Haiku:** label=mischaracterization, confidence=0.75
  - reason: The visible portion of the woman extends from approximately the chin/lower face down to mid-thigh or knee area, not specifically 'mid-neck down to just above the knees'.
- **Sonnet:** verdict=reject, confidence=0.7
  - reason: The image shows the woman from roughly the chin/neck down to about mid-thigh/knee level, which reasonably matches 'mid-neck down to just above the knees'.

### 30. train-en-1339 (dev, known-TP)
- **Flagged span:** 'handbag, which is tucked in under her arm'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The girl in the foreground is not carrying a handbag under her arm; she is holding a memorial sign with both hands in front of her body.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The woman in front has a black bag hanging on her shoulder/side, but she is holding the memorial sign with both hands, not carrying a handbag tucked under her arm.

### 31. train-en-1767 (dev, known-TP)
- **Flagged span:** 'red beak'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The swan's beak is pink/pale colored, not red as stated in the response.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The bird's beak is mostly pale grayish-pink with a red knob at the base near the face, so calling the beak itself 'red' mischaracterizes it.

### 32. train-en-683 (dev, known-TP)
- **Flagged span:** 'white or pale yellow bands'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The bands are distinctly white, not pale yellow; describing them as 'pale yellow' mischaracterizes their actual color.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The bands appear whitish with a subtle cream/pale tone in the lighting, so describing them as 'white or pale yellow' is a reasonable hedge rather than a clear mischaracterization.

### 33. train-en-1517 (dev, known-TP)
- **Flagged span:** 'no red car visible'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: There is a red car visible in the background on the right side of the image, partially visible next to other vehicles.
- **Sonnet:** verdict=confirm, confidence=0.98
  - reason: A red car is clearly visible in the background to the right of the blue GMC SUV.

### 34. train-en-11398 (dev, known-TP)
- **Flagged span:** 'pinkish'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The shrimp in the image display predominantly grayish, greenish, and purplish tones with red/burgundy accents, not pinkish coloring.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The shrimp in the image are raw and display grayish, greenish, and purplish tones with dark banding, not pinkish coloring, which is characteristic of cooked shrimp.

### 35. train-en-11309 (dev, known-TP)
- **Flagged span:** 'cucumbers'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The image shows potatoes, not cucumbers. The round, starchy vegetables with small dark speckles and yellowish skin are characteristic of yellow potatoes, not cucumbers.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The round yellow fruits with visible stem/blossom scars and speckled skin are consistent with lemon cucumbers, a round yellow cucumber variety, not potatoes.

### 36. train-en-1538 (dev, known-TP)
- **Flagged span:** 'right hand'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The man is holding the phone in his left hand, not his right hand, based on the visible positioning in the image.
- **Sonnet:** verdict=PARSE_ERROR, confidence=None

### 37. train-en-1274 (dev, known-TP)
- **Flagged span:** 'light-colored stripe running across the forehead'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The distinctive light coloration on the head appears primarily on the crown/top of the head and extends back, not as a stripe running across the forehead as described.
- **Sonnet:** verdict=confirm, confidence=0.72
  - reason: The pale coloration is concentrated on the crown/top of the head and upper face, not as a stripe crossing the forehead specifically.

### 38. train-en-995 (dev, known-TP)
- **Flagged span:** 'plastic'
- **Haiku:** label=mischaracterization, confidence=0.75
  - reason: The pink object appears to be made of fuzzy or textured material (possibly plush or fabric-like), not smooth plastic as claimed.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The pink object has a textured, molded appearance consistent with plastic (similar to other plastic baskets/trays in the background), not fabric or plush material.

### 39. train-en-477 (held-out, known-FP)
- **Flagged span:** 'lighter, almost white or lavender stripes'
- **Haiku:** label=mischaracterization, confidence=0.9
  - reason: The eggplant does not have visible white or lavender stripes; the surface is predominantly dark purple with subtle ribbing/ridges but no distinct striping pattern.
- **Sonnet:** verdict=reject, confidence=0.85
  - reason: The eggplant clearly shows lighter, whitish-lavender streaks radiating across its purple surface, visible in the image.

### 40. train-en-2371 (held-out, known-FP)
- **Flagged span:** 'cracked asphalt or concrete'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The surface surrounding the drain is cracked concrete/cement, not asphalt. The texture and appearance clearly show a concrete material, not asphalt.
- **Sonnet:** verdict=reject, confidence=0.6
  - reason: The surrounding surface's texture with embedded aggregate and cracking pattern is ambiguous and could plausibly be asphalt, so the original hedge 'asphalt or concrete' was reasonable rather than clearly wrong.

### 41. train-en-2057 (held-out, known-FP)
- **Flagged span:** 'propellers'
- **Haiku:** label=mischaracterization, confidence=0.9
  - reason: The aircraft has ducted fans, not traditional propellers; the rotors are enclosed within circular ducts/rings, not open propellers.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The rotors are essentially ducted propellers, so calling them 'propellers' is not technically wrong even though 'ducted fans' is a more precise description.

### 42. train-en-1828 (held-out, known-FP)
- **Flagged span:** 'The yellow and teal containers hold whole coconut fruits that still possess their hard outer shell'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The yellow container on the left holds what appears to be a whole coconut with shell, but the teal container on the right clearly contains deshelled/processed coconut meat (white flesh and brown pieces), not whole fruits with shells.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The teal container shows what appears to be dehusked white coconut flesh and brown pieces, not whole coconuts with intact shells, contradicting the claim that both containers hold whole shelled fruits.

### 43. train-en-11101 (held-out, known-FP)
- **Flagged span:** 'blue SUV'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The vehicle is a compact utility vehicle (appears to be a Lada Niva or similar), not an SUV, and it is dark blue/teal, not bright blue.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The vehicle (a Lada Niva) is a compact 4x4 which is reasonably classified as an SUV, and its dark teal/blue-gray color can fairly be described as blue.

### 44. train-en-830 (held-out, known-FP)
- **Flagged span:** 'bright'
- **Haiku:** label=mischaracterization, confidence=0.75
  - reason: The star visible in the upper center of the image is very small and faint, not notably bright or prominent against the sky.
- **Sonnet:** verdict=reject, confidence=0.6
  - reason: The star appears as a distinct bright white point clearly standing out against the dark sky, justifying the description as bright.

### 45. train-en-469 (held-out, known-FP)
- **Flagged span:** 'does not appear to have stitching'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The football clearly displays visible curved stitching lines (the teal/blue curved seams) running diagonally across the yellow surface, which contradicts the claim of no stitching.
- **Sonnet:** verdict=reject, confidence=0.93
  - reason: The teal curved lines are painted/molded color patterns, not actual stitching or seams, so the response's claim of no stitching is correct.

### 46. train-en-876 (held-out, known-TP)
- **Flagged span:** 'still has its skin on'
- **Haiku:** label=mischaracterization, confidence=0.95
  - reason: The potato in the hand is clearly peeled, showing only pale yellow flesh with no brown skin visible on the surface being held.
- **Sonnet:** verdict=confirm, confidence=0.97
  - reason: The potato held in the hand is fully peeled, showing pale yellow flesh with no brown skin remaining, contradicting the response's claim.

### 47. train-en-1483 (held-out, known-TP)
- **Flagged span:** 'rectangular blade'
- **Haiku:** label=mischaracterization, confidence=0.9
  - reason: The grater drums visible in the image have cylindrical shapes with small holes arranged in patterns, not rectangular blades; the drums are circular/cylindrical, not rectangular.
- **Sonnet:** verdict=confirm, confidence=0.9
  - reason: The drums shown are cylindrical with round or slotted holes, not rectangular blades, so describing them as a 'rectangular blade' mischaracterizes their shape.

### 48. train-en-11139 (held-out, known-TP)
- **Flagged span:** 'rose or a floral pattern'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The carved design on the spoon handle appears to be a decorative geometric or abstract pattern, not a rose or floral design.
- **Sonnet:** verdict=reject, confidence=0.75
  - reason: The lower part of the utensil handle visible in the image shows an embossed floral/rose-like design, so the claim is reasonably accurate.

### 49. train-en-2207 (held-out, known-TP)
- **Flagged span:** 'grayish areas around its neck'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The bird displays distinct black or dark markings on its neck and upper body, not grayish areas; the coloration is black and white, not gray.
- **Sonnet:** verdict=confirm, confidence=0.8
  - reason: The neck and upper back markings appear as distinct black/dark patches contrasting with white, not grayish tones.

### 50. train-en-1727 (held-out, known-TP)
- **Flagged span:** 'suspended from the ceiling by chains or straps'
- **Haiku:** label=mischaracterization, confidence=0.9
  - reason: The chair is suspended by a single chain/hook from above, not by multiple chains or straps as stated.
- **Sonnet:** verdict=confirm, confidence=0.85
  - reason: The image shows the chair suspended by a single chain from a curved stand, not multiple chains or straps from a ceiling.

### 51. train-en-9671 (held-out, known-TP)
- **Flagged span:** 'blue and red text'
- **Haiku:** label=mischaracterization, confidence=0.85
  - reason: The banner visible in the image has blue text on a white background, not blue and red text together on the sign itself.
- **Sonnet:** verdict=reject, confidence=0.97
  - reason: The white sign held by the man clearly shows both blue text (top lines) and red text (bottom line), so the claim of blue and red text is accurate.

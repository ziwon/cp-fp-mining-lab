# False Positive Taxonomy

A clear false-positive taxonomy is more useful than a single `false_positive` label.

## Fire

- headlight
- reflection
- welding
- sunset
- warning_light
- screen_display

## Smoke

- steam
- fog
- dust
- cloud
- exhaust_gas
- lens_blur

## Falldown

- sitting
- lying_object
- crouching
- shadow
- occlusion
- low_resolution

## Intrusion

- animal
- tree_motion
- camera_noise
- reflection
- authorized_worker

## Recommended review fields

```text
is_event:
  real_event / false_positive / uncertain

fp_type:
  steam / fog / dust / reflection / headlight / shadow / sitting / animal / unknown

bbox_valid:
  valid / wrong_class / wrong_location / unnecessary

comment:
  free text root-cause note
```

# Race Variables

Oct 11, 2026

Every variable that shapes a race result, grouped by where it comes from. For each: what it does to results, how we can measure it from public data, and how it enters the model.

## Summary

A driver's finishing position comes down to four things: where they start, how fast they are over a race distance, how much track position they gain or lose, and what random events happen. About 75 variables sit underneath those four, grouped below by where they come from.

&#91;embedded content: driver tree · finishing position and its causes\]

The car sets the window of possible results, the driver places the result within it, and the circuit decides how much starting position locks things in. Weather and race events are the main sources of upsets.

## 1. Car performance

The car is the single biggest factor in modern F1: it sets the window a driver can finish in. It is also the variable that moves most during a season, so it needs a time-varying estimate, not a fixed rating.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Underlying team pace | Sets the baseline lap time for both cars | Fuel- and tyre-corrected race laps; qualifying gap to pole | Team effect in the pace model, a random walk across races (Phase 3) |
| Development rate / upgrades | Pace shifts between races; teams diverge over a season | Change in corrected pace race to race; upgrade lists teams publish each weekend | Drift in the random walk; optional "upgrade this weekend" flag |
| Car–circuit fit (downforce level) | Some cars excel on high-speed or low-speed tracks | Pace residuals grouped by circuit type; speed-trap vs sector times | Team × circuit-cluster interaction |
| Straight-line speed / drag | Defends and attacks on straights; matters more where overtaking is hard | Speed traps, top speed from telemetry | Input to the overtaking model |
| Power unit (2026: 50/50 combustion and electric, no MGU-H, 350 kW MGU-K) | Differences between engine makers; energy deployment shapes lap time | Pace grouped by engine supplier; telemetry speed profiles on straights | Engine-supplier effect nested above team |
| Energy management (2026) | Running out of deployable energy costs time on straights; harvesting choices matter | Speed drop-off at the end of straights ("clipping") in telemetry | Circuit × engine term; feature for straight-heavy tracks |
| Reliability | DNFs and grid penalties for exceeding component allocations | Retirement status codes (Jolpica); grid penalties in race-control messages | DNF hazard per team-season (Phase 6) |

For 2026, the new regulations target a roughly even split between combustion and electric power, drop the MGU-H, and raise MGU-K output from 120 kW to 350 kW ([GP247](https://www.grandprix247.com/2025/12/17/f1-news-boost-overtake-modes-2026-formula-1-terminology-explained-2026-regulations/)). That makes energy deployment a new performance variable that pre-2026 data cannot teach.

## 2. Driver

The driver decides where in the car's window a result lands. The teammate is the only clean comparison, because they share the same car, so most driver variables are measured as gaps to the teammate.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Raw pace | Lap-time advantage over the teammate | Teammate gap in qualifying and on clean race laps | Driver effect in the pace model, partially pooled across seasons |
| Qualifying vs race pace | Some drivers out-qualify their race pace, others the reverse | Separate teammate gaps in qualifying and in the race | Correlated quali and race effects per driver |
| Racecraft (attack and defence) | Positions gained or lost in wheel-to-wheel fights | On-track passes made and conceded, adjusted for pace gap (position data, excluding pit cycles) | Driver term in the overtaking model |
| Tyre management | Slower degradation allows longer stints and different strategies | Degradation slope vs the teammate on the same compound | Driver modifier on the degradation curve |
| Starts and lap 1 | Places gained or lost before the race settles | Position at end of lap 1 vs grid | Driver term in the lap 1 model |
| Consistency | Fewer slow laps and mistakes | Spread of clean-lap times after correction | Driver-level noise in lap-time sampling |
| Error / crash rate | DNFs and incidents caused by the driver | Retirements coded as accident or collision, attributed to the driver | Driver crash hazard, heavily shrunk toward the field average |
| Wet-weather skill | Skill gaps widen in the rain | Teammate gaps in wet sessions only (sparse) | Driver × wet interaction, with a wide prior |
| Experience | Rookies improve through a season; circuit familiarity helps | Races started; previous starts at that circuit | Rookie learning-curve term |
| Energy deployment skill (2026) | Better use of boost, Overtake Mode and recharge | Teammate gaps on straights from telemetry; still an open research question | Experimental feature; test in Phase 7 |

## 3. Tyres

Tyres turn pace into strategy. The interaction of compound, age and circuit decides how many stops a race needs and who gains from them.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Compound choice for the weekend | Pirelli picks three dry compounds per event; softer is faster but wears sooner | Compound per stint (FastF1) | Categorical input to pace and degradation |
| Degradation rate | Lap-time loss per lap of tyre age; sets stint length | Slope of corrected lap time against tyre age | Degradation curve per compound × circuit (Phase 4) |
| The "cliff" | Sudden pace loss late in a stint forces a stop | Breakpoints in the lap time vs age curve | Spline term in the degradation model |
| Warm-up | Hard tyres can be slow for a lap or two, which matters at restarts and after stops | Out-lap and first-laps pace after a stop | Warm-up penalty on the first laps of a stint |
| Graining and overheating | Temporary or permanent pace loss, strongest in hot or cool extremes | Indirectly: residuals by track temperature | Temperature interaction on degradation |
| Tyre sets left for the race | A driver with new sets has more strategy options | Sets used in practice and qualifying (stint data) | Constraint on the strategy generator |
| Starting tyre | Determines first stint length and strategy | Compound at race start | Input to the strategy chooser |
| Mandatory compound rule | Forces at least one stop in a dry race | Season rules | Rules config (`configs/rules/*.yaml`) |

## 4. Strategy and team operations

The pit wall and the pit crew can win or lose places without the car changing at all. These variables are team-level and fairly stable within a season.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Pit stop execution | A slow stop loses places directly | Stationary and pit-lane times (Jolpica pit stops, FastF1 pit in/out) | Stop-time distribution per team, with a slow-stop tail |
| Undercut / overcut | Stopping earlier or later than a rival to gain track position | Position changes across pit cycles; out-lap vs in-lap pace | Emerges in the simulator from pace, warm-up and pit loss |
| Strategy calls | Choosing the stop count and timing; reacting to safety cars | Actual strategies vs the time-optimal plan | Team tendency term in the strategy chooser |
| Team orders | Positions swapped between teammates | Teammate swaps without a pace reason; race-control and radio context | Optional rule in the simulator (a lead driver per team) |
| Unsafe releases and pit errors | Time penalties, lost wheels, DNFs | Race-control messages | Small hazard per pit stop |
| Penalties during the race | 5/10-second penalties, drive-throughs | Race-control messages | Penalty hazard per driver per race |
| Grid penalties | Component changes push drivers back | Starting grid vs qualifying position | Known before the race: applied to the starting grid |

## 5. Circuit

The circuit decides how much the other variables matter. On a track where passing is hard, qualifying and strategy dominate; where it's easy, race pace wins out.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Overtaking difficulty | How strongly grid position predicts the result | Historical on-track passes per race, adjusted for pace spread | Circuit term in the overtaking model |
| Layout type (speed profile) | Favours cars strong in slow corners, fast corners or on straights | Share of the lap at full throttle; corner speed mix from telemetry | Circuit clusters for team × circuit interactions |
| Pit-lane time loss | Sets the cost of each stop, so the number of stops | Pit-in plus pit-out laps against a normal lap | Per-circuit pit loss (Phase 4) |
| Surface abrasiveness | Drives tyre degradation and the stop count | Observed degradation rates at the circuit | Circuit effect in the degradation model |
| Track evolution | Grip improves as rubber goes down; quali and race laps get faster | Lap-time trend through sessions | Track-evolution term when correcting pace |
| Street circuit / walls | More safety cars and crashes; higher DNF risk | History of track-status interruptions per circuit | Higher safety-car and crash hazard |
| Distance to first corner | Affects lap 1 shuffle and first-corner incidents | Circuit geometry (FastF1 circuit info) | Lap 1 model input |
| Altitude (e.g. Mexico City) | Thinner air: less downforce and cooling, changes engine behaviour | Circuit elevation | Circuit covariate; flag for PU-sensitive tracks |
| Energy recharge profile (2026) | Tracks differ in how much energy can be recovered, changing deployment | Recharge allowance per circuit in the regulations; telemetry | Circuit × engine term |
| New or changed circuit | No history to learn from | Calendar | Fall back to circuit-cluster priors |

## 6. Weather and environment

Weather is the variable that most often overturns a forecast. Rain changes the pace order, the strategy and the incident rate at once.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Rain (before or during the race) | Reshuffles the order; more incidents and safety cars; intermediate/wet tyre calls become decisive | FastF1 weather (rainfall flag) and wet-tyre usage | Wet regime: wider lap-time noise, driver × wet effect, higher hazards |
| Changing conditions (drying track) | Timing of the switch between wet and dry tyres wins and loses races | Lap-by-lap compound changes during mixed races | Simulator state for track wetness; a stretch goal |
| Track temperature | Shifts degradation and which cars "switch on" their tyres | FastF1 track temperature | Interaction with degradation and team pace |
| Air temperature and humidity | Small effects on engine power and cooling | FastF1 weather data | Minor covariates; test before keeping |
| Wind | Changes balance in corners and top speed on straights | FastF1 wind speed and direction | Mostly noise; useful only at exposed circuits |
| Night or twilight race | Track cools during the race, so conditions differ from daytime practice | Session start time | Adjust practice-to-race pace transfer |
| Forecast uncertainty | Predictions are made before race day | Forecast probability of rain at prediction time | Mix dry and wet simulations by rain probability |

## 7. Grid and race events

Grid position is the strongest single predictor available before a race. Race events are the randomness on top of it, and the simulator should sample them, not average them away.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Grid position | Track position is hard to win back, especially where passing is difficult | Starting grid (after penalties) | Starting state of every simulation |
| Clean vs dirty side of the grid | Grip difference off the line at some circuits | Lap 1 gains by grid side, per circuit | Small lap 1 term |
| Pit-lane start | Starts behind everyone, but can change setup and tyres | Starting grid | Starting state |
| Lap 1 incidents | Contact and damage in the first corners | Retirements and damage on lap 1; race-control messages | Lap 1 incident probability, higher in the midfield pack |
| Safety car | Bunches the field, cuts the gap, makes stops cheaper; can make or ruin strategies | Track-status data per lap | Per-circuit hazard; SC effects in the simulator (Phase 6) |
| Virtual safety car | Slows everyone; also cheapens stops, less than a full SC | Track-status data | Separate hazard, same mechanics |
| Red flag | Free tyre change, and a standing restart (a second "lap 1") | Track-status data | Rare-event hazard; restart reuses the lap 1 model |
| Mechanical DNF | Removes a car from the result entirely | Retirement status codes | Team reliability hazard (from section 1) |
| Accident DNF | Same, driver-caused | Retirement status codes | Driver crash hazard (from section 2) |
| Post-race disqualification | Rare; changes the official result | Final classification vs finishing order | Ignore for prediction; score against the official result |

## 8. In-race dynamics

These are the interactions between cars on track. They are why a faster car doesn't simply finish ahead, and they are the hardest part of the simulator to get right.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Dirty air | A following car loses downforce and wears its tyres faster | Lap-time loss when within about a second of the car ahead (gap data) | Following penalty in the simulator |
| Overtake aid (DRS until 2025; Overtake Mode from 2026) | Makes passing easier on straights | Pass rates with vs without the aid; the 2026 change resets this | Era-specific term in the overtaking model |
| Active aero (2026) | All cars switch wings between Corner Mode and Straight Mode, changing straight-line speed | Telemetry speed profiles | Part of the 2026 car-performance baseline |
| Traffic and backmarkers | Time lost behind slower cars and when lapping | Lap-time loss near lapped cars; blue flags in race-control messages | Traffic penalty in the simulator |
| Fuel load | A heavy car is slower early; lap times fall as fuel burns | Lap-time trend through the race | Fuel correction in the pace model |
| Fuel and energy saving | Lift-and-coast and recharging cost lap time but protect energy and fuel | Telemetry: throttle lifts before braking zones | Absorbed in race pace; explicit only if telemetry shows it matters |
| Defending | Defending costs both cars time, so the pack behind closes in | Lap-time loss for both cars during fights | Battle penalty for both cars in the simulator |

On the 2026 change: DRS has been replaced by Active Aero, with front and rear wings switching between modes in activation zones, and by Overtake Mode, which lets the chasing car deploy the full 350 kW to a higher speed than the car ahead ([The Race](https://www.the-race.com/formula-1/boost-overtake-mode-active-aero-recharge-key-2026-terms-explained)). Pass rates learned before 2026 should not be reused unchanged.

## 9. Weekend format and wider context

These don't change lap times directly. They change how much we know going into the race, or what drivers and teams are trying to achieve.

| Variable | Effect on results | How we measure it | Where it enters the model |
| --- | --- | --- | --- |
| Sprint weekend | Only one practice session, so teams arrive less prepared; the sprint itself gives race-pace data | Event format (FastF1 schedule) | Wider pace uncertainty; sprint laps as extra evidence |
| Lost practice running | A driver with little practice has a poorer setup and less data | Laps completed per driver per session | Wider uncertainty for that driver |
| Regulation era | Resets the pecking order (2014, 2022, 2026) | Season | Prior reset strategies (Phase 7) |
| Budget cap and development timing | Late-season upgrades are scarce; some teams switch to next year's car | Pace drift late in a season | Slows the random walk late in the year |
| Championship situation | A title contender may race conservatively; team orders become likely | Standings before the race | Optional feature; test before keeping |
| Home race / driver contract news | Often discussed, rarely measurable | — | Leave out; treat as noise |
| Fatigue, illness, travel | Real but invisible in public data | — | Leave out |
| Driver changes mid-season | A substitute or new driver has no history in that car | Entry list | Rookie prior; team effect carries over |

The last few rows matter as a principle: an honest model leaves out what it can't measure, and lets the uncertainty show it.

## 10. What the data gives us, and what stays hidden

Most of the variables above can be estimated from public data, but several of the most important ones only indirectly. Fuel load, setup and energy state are hidden, so the model has to infer them from lap times and telemetry.

| Source | What it provides | Coverage |
| --- | --- | --- |
| [FastF1](https://docs.fastf1.dev/data_reference/index.html) timing | Lap and sector times, speed traps, compounds, tyre age, stints, pit in/out | 2018 onwards |
| FastF1 telemetry | Speed, throttle, brake, gear, RPM; car position on track | 2018 onwards |
| FastF1 session data | Weather, track status (SC, VSC, red flag), race-control messages | 2018 onwards |
| [Jolpica-F1](https://docs.fastf1.dev/api_reference/jolpica.html) | Results, grids, qualifying, pit stops, retirement status, standings | 1950 onwards (detail varies by era) |
| OpenF1 | Live-timing data including intervals and team radio | Recent seasons; worth evaluating as a second source |

**Not in any public source:** fuel loads, car setup, battery state and deployment maps, tyre temperatures and pressures, damage, team strategy intentions. Treat these as latent variables: their effects show up in lap times, so the model absorbs them as noise or infers them (for example, fuel from the lap-time trend).

## 11. What to model first

Start with the variables that matter most and that public data measures well; they carry the MVP (Phases 3–5). The high-impact but hard-to-measure ones come next (Phase 6), because they drive upsets.

&#91;embedded content: 2x2 · impact against measurability\]

Everything in "Add if it improves the backtest" earns its place only through the Phase 6 ablation table; everything in "Leave out" stays out unless new data appears.

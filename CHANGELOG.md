# Changelog

## [0.3.0](https://github.com/Tachy/World-to-BeamNG/compare/v0.2.0...v0.3.0) (2026-09-28)


### Features

* Add a --loglevel option to world_to_beamng.py ([7efdc86](https://github.com/Tachy/World-to-BeamNG/commit/7efdc86db0b96234d9ab2bdce43975b0406ea536))
* Add corner radii per road class for junction fillets ([a9fadbc](https://github.com/Tachy/World-to-BeamNG/commit/a9fadbc0b6ea31cdcd201eaf0a0fa146053d4c65))
* Build the fill mesh of rounded junction corners ([2f443cc](https://github.com/Tachy/World-to-BeamNG/commit/2f443cc4b15cef91a753238bc77f23f9f18ce05e))
* Build the kerb and sidewalk cross-section mesh ([0f9bc26](https://github.com/Tachy/World-to-BeamNG/commit/0f9bc26e49b60a23fd89fc2245980f5d8f8c1cd6))
* Drape junction fills onto the terrain and drop their collision ([3c7bd83](https://github.com/Tachy/World-to-BeamNG/commit/3c7bd8330a6819b0399dcef86ce7e6a863aedad9))
* Embed rounded junction corners into the terrain ([7726b44](https://github.com/Tachy/World-to-BeamNG/commit/7726b44bdfe40c685bc8e202dc6e46b9e7513437))
* Export kerbs and raised sidewalks along tagged roads ([8dea892](https://github.com/Tachy/World-to-BeamNG/commit/8dea892bcf5aabd9d725cc11df74157af44a8f08))
* Export rounded junction corners and run sidewalks around them ([2158a0e](https://github.com/Tachy/World-to-BeamNG/commit/2158a0efb443d524565864f3d0f7df737cf653db))
* Find junction corners and fit tangent-circle fillets ([d49fe75](https://github.com/Tachy/World-to-BeamNG/commit/d49fe7514e8e850e6f91578168380bf0ead82ea9))
* Leave junction corners of 140 degrees and more without fill ([c839a06](https://github.com/Tachy/World-to-BeamNG/commit/c839a06efa18eba16ca86003a6be9b6d26f9d8d2))
* Map sidewalk surfaces to road surface types ([3119f7c](https://github.com/Tachy/World-to-BeamNG/commit/3119f7c555f378fabd1d4eb34ccad56ca360778c))
* Plan sidewalk kerb lines with gaps at joining roads ([b10be51](https://github.com/Tachy/World-to-BeamNG/commit/b10be518ea624e6b675e266ba71ce0c6dde98446))
* Read sidewalk sides and surfaces from OSM road tags ([90eac1d](https://github.com/Tachy/World-to-BeamNG/commit/90eac1dd59f33678a14d7d6d85412cd10c7a94b1))
* Round gravel track corners and fill with the joining road's surface ([8483ea5](https://github.com/Tachy/World-to-BeamNG/commit/8483ea564d27ea08f60445112238cfad6dfc12d2))
* Run sidewalks around rounded junction corners ([082b1f3](https://github.com/Tachy/World-to-BeamNG/commit/082b1f341a02fe3a94985c64c0d4929a920a0039))
* Shrink junction radii step by step where the full radius does not fit ([f1c7551](https://github.com/Tachy/World-to-BeamNG/commit/f1c7551fbbf2fe93c0623d0f6873c5cb0c0f2736))
* Start the road embankment behind the sidewalk ([aa87b50](https://github.com/Tachy/World-to-BeamNG/commit/aa87b5053b0b25bbbb02796ae3a5b9267b299c21))
* Tile BeamNG's stock asphalt to its real grain size on roads ([35ff969](https://github.com/Tachy/World-to-BeamNG/commit/35ff9693e833349352ddc0e81ad92b5778bb2ae7))
* Use BeamNG's homogeneous tileable asphalt for roads ([c3fbe0c](https://github.com/Tachy/World-to-BeamNG/commit/c3fbe0c83e6ebe22c96e9cfb074e7ae2ed2c38ea))
* Widen the road embedding behind tagged sidewalks ([78d768e](https://github.com/Tachy/World-to-BeamNG/commit/78d768e9fb05645359871da85bb3a6c6d701346e))


### Bug Fixes

* Blend corner heights, fit fillets to curved kerbs, embed past the arc, join sidewalks order-independently ([fb96b02](https://github.com/Tachy/World-to-BeamNG/commit/fb96b02ee9b60e37f51ce7206d7ae02b1e0c1277))
* Draw no junction fill from 160 degrees opening angle on ([da031e1](https://github.com/Tachy/World-to-BeamNG/commit/da031e122fb89d2d0ec43a635cb41582443f1ff6))
* Fit junction fills to curved arms and level them to the roads ([7c91719](https://github.com/Tachy/World-to-BeamNG/commit/7c91719f815031163fa3fd61ade0180a326f41d1))
* Keep junction corners whose kerb offset GEOS splits into touching pieces ([b49696c](https://github.com/Tachy/World-to-BeamNG/commit/b49696cf13aa8d0e7c0bc2685f976f49a7a55739))
* Keep sidewalks through street continuations and off roads without decals ([16aeb7b](https://github.com/Tachy/World-to-BeamNG/commit/16aeb7bddb728503fc8e4bdf3f80b70352fd0e43))
* Round only the tip of acute junction corners ([53ceb35](https://github.com/Tachy/World-to-BeamNG/commit/53ceb35bf49b1500d114150a4ae90c62dec270f3))

## [0.2.0](https://github.com/Tachy/World-to-BeamNG/compare/v0.1.0...v0.2.0) (2026-09-27)


### Features

* End every gallery column row with a flush column at both ends ([73bb966](https://github.com/Tachy/World-to-BeamNG/commit/73bb966098416aa0848bbf82aa711ede138d3fcb))
* Leave dirt tracks and footways without DecalRoads ([86a1d8b](https://github.com/Tachy/World-to-BeamNG/commit/86a1d8b62531ad0696d22755d064d5b161dc68af))
* Make the bridge retouch of the aerial photo switchable, off by default ([be4fdbb](https://github.com/Tachy/World-to-BeamNG/commit/be4fdbb71d364f82aba16752b2a4aa3343ba4721))
* Read loose CityGML files as building source ([209be33](https://github.com/Tachy/World-to-BeamNG/commit/209be3342abcd19655c8b8f59e5147b779dc3fc5))
* Read orthophotos as JPG/PNG with any world file ([ef912da](https://github.com/Tachy/World-to-BeamNG/commit/ef912da6b64f26f5f6b318f2eca577ebd7fd7dbd))
* Read swissBUILDINGS3D 2.0 DXF buildings next to LoD2 CityGML ([20ed4b2](https://github.com/Tachy/World-to-BeamNG/commit/20ed4b2d24cfde2c053550ed6f8adf9248fe5d5a))
* Use an ambientCG PBR texture for asphalt roads ([83fc366](https://github.com/Tachy/World-to-BeamNG/commit/83fc366f08e51601de69363a92bef23468edf33c))


### Bug Fixes

* Keep lane split branches smooth outside a corner of the main axis ([f8b4b71](https://github.com/Tachy/World-to-BeamNG/commit/f8b4b715987d1937d35b32ed697cb075da413cad))
* Keep the DGM30 tile cache per target CRS ([7174825](https://github.com/Tachy/World-to-BeamNG/commit/71748252b77e29695b3a74c9b4f95d0eeeb6e599))
* Let two lane splits share a link between them ([731f58a](https://github.com/Tachy/World-to-BeamNG/commit/731f58a6931a1be0283a0153196c056f13f91890))
* Make the asphalt strip square and use BeamNG's asphalt by default ([9e47bd4](https://github.com/Tachy/World-to-BeamNG/commit/9e47bd48184eba419f818fa9d4203ea4b25732e0))
* Place world-file aerial photos on their pixel corners ([5994a09](https://github.com/Tachy/World-to-BeamNG/commit/5994a09aa68f0e2d492472fd258d8536204f90be))
* Spread road decals over render priorities so BeamNG draws them all ([3814873](https://github.com/Tachy/World-to-BeamNG/commit/381487350eb25d1cae09b93130c9202e09fdacb1))
* Stop DGM30 tile seams from digging trenches into the horizon ([e3a137e](https://github.com/Tachy/World-to-BeamNG/commit/e3a137e22e7219d8fa0806548d5bec6bb3881d3d))


### Refactoring

* Move terrain workflow helpers into their own modules ([58e9525](https://github.com/Tachy/World-to-BeamNG/commit/58e9525dec883706e1d6707bc1a40549db1e330a))
* Remove dead gallery helper and unused locals ([1d6c3c3](https://github.com/Tachy/World-to-BeamNG/commit/1d6c3c335bc945ee0bcaff4f88dd316542ab5a5f))
* Share arc length, smoothstep and point helpers ([1e697c4](https://github.com/Tachy/World-to-BeamNG/commit/1e697c47aa11a6c70baf4effefe1c9c46f27427d))
* Split BeamNGExporter.export_complete_level into steps ([2126aef](https://github.com/Tachy/World-to-BeamNG/commit/2126aefaf04e108b37f4d7858a97d56e895629eb))
* Split TerrainWorkflow.process_tile into phases ([a842835](https://github.com/Tachy/World-to-BeamNG/commit/a842835b947efe43ccf81e8895743ac2c413a1b4))

## [0.1.0](https://github.com/Tachy/World-to-BeamNG/compare/v0.0.1...v0.1.0) (2026-09-26)


### Features

* Align road lines with the structure's 50 m before a bridge, tunnel or gallery ([aa86a6f](https://github.com/Tachy/World-to-BeamNG/commit/aa86a6f20c5d0fbe95274614ad2cb63ed3fce03e))
* Block stripes replace the dashed divider along the taper zone of a dropped or added lane ([4b5f6d0](https://github.com/Tachy/World-to-BeamNG/commit/4b5f6d082c70006b2a2f05a56bebe4a37904c679))
* Bridges follow width transitions instead of keeping a fixed width ([1f5520c](https://github.com/Tachy/World-to-BeamNG/commit/1f5520c3140257be4205fe115a5b6e373308c819))
* Carry the trunk's deck on past a split and cut it into the branches ([996f52e](https://github.com/Tachy/World-to-BeamNG/commit/996f52e7c3260181fafe2f7c916c6e99076477e7))
* Centre line of a 2-lane road runs onto the double line of the wider road ([2ec3166](https://github.com/Tachy/World-to-BeamNG/commit/2ec316608311d0e404ec02c531e4f66d5e5bc078))
* Continue the double line on the two-lane road after a change to more lanes ([cb66ea7](https://github.com/Tachy/World-to-BeamNG/commit/cb66ea7d84d55834e63beda46fd0eee2a487f99d))
* Double centre line in two-lane tunnels and galleries ([f5e10f8](https://github.com/Tachy/World-to-BeamNG/commit/f5e10f8566dd8eb52ecdffd7076a10bbc8d9c75e))
* Double the tunnel light brightness and add visible glowing lamp bodies ([3c68702](https://github.com/Tachy/World-to-BeamNG/commit/3c68702081884767d0f31bf9294b0086eb205219))
* End dashed lane dividers 50 m before a structure that has none ([271b8a3](https://github.com/Tachy/World-to-BeamNG/commit/271b8a38ed75cc25fe6c7d7c7c5d08fca041a1b0))
* Gallery plinth and bridge curbs stand outside the carriageway ([1a9e2b1](https://github.com/Tachy/World-to-BeamNG/commit/1a9e2b14cf8819a1833f5ec3247a0a8b98b73ea7))
* Guard rails along roads with a drop beside them ([200ab87](https://github.com/Tachy/World-to-BeamNG/commit/200ab87872320ac484474aa4a5df71443e5fa6bb))
* Keep split branches in their lanes and share one bridge deck ([75a082d](https://github.com/Tachy/World-to-BeamNG/commit/75a082d22de6cb625bd14280d7621ec206c750c4))
* Keep the uninvolved lane at a constant width through a lane taper ([7d0351d](https://github.com/Tachy/World-to-BeamNG/commit/7d0351d837c2a0e262f93311d3aab1afeb085280))
* Light tunnels with ceiling spot lights like the vanilla italy tunnel ([1cca672](https://github.com/Tachy/World-to-BeamNG/commit/1cca672fb80a9137be2f1d74142e9661065de6a1))
* Longer road width transitions for lane changes and at structures ([21f501d](https://github.com/Tachy/World-to-BeamNG/commit/21f501dad1b02d677aad414018d0696adc43e262))
* Make bridge piers half the carriageway width across and half of that thick ([c6ae408](https://github.com/Tachy/World-to-BeamNG/commit/c6ae4083aebd69edd4ded470b7dd1e6134d6cd9e))
* Make bridge piers wide walls that reach 5 m into the ground ([a09c3cd](https://github.com/Tachy/World-to-BeamNG/commit/a09c3cdc5b8c5fc7775c39379583ac4d7e46b066))
* Mark the lane boundaries on the stem of a lane split ([fe02fd9](https://github.com/Tachy/World-to-BeamNG/commit/fe02fd922c1f4640b7adcf745acebd59a412bce8))
* Retouch bridge decks out of the aerial photo ([eac364a](https://github.com/Tachy/World-to-BeamNG/commit/eac364a2dbd95ee4c2ac1cdfb578cc55a4d4cbc5))
* Roads on structures with mesh markings, decal-like floor UVs and an invisible AI road ([2f729e1](https://github.com/Tachy/World-to-BeamNG/commit/2f729e1944a3f0cccd7e7ec3c24c085b0f7118bd))
* Smooth split branches and keep split bridges linear in height ([a4b3453](https://github.com/Tachy/World-to-BeamNG/commit/a4b345338bbe1dc78a2ad695d0a5653e24a4293f))
* Solid double centre line on two-way roads with three or more lanes ([bb05e92](https://github.com/Tachy/World-to-BeamNG/commit/bb05e92177c598ba883b55b0f2c252d653fb2cab))
* Split the trunk's lanes onto the branches at motorway exits ([54edd69](https://github.com/Tachy/World-to-BeamNG/commit/54edd696466d41db2d957df15edb8ad6ab92809b))
* Tunnel curbs and a tube profile with 4.20 m above the carriageway edge ([ffb882f](https://github.com/Tachy/World-to-BeamNG/commit/ffb882facf3888bdd92f3305ecdb966f811f1a19))
* Tunnel darkness starts 50 m behind open portals ([7171827](https://github.com/Tachy/World-to-BeamNG/commit/717182706ec101bcf6f85312e7828b6b73b4e9b8))


### Bug Fixes

* Close the seams where a split bridge deck changes parts ([ebfab81](https://github.com/Tachy/World-to-BeamNG/commit/ebfab81d33d81ecdc08895d31cf699cee8fa26aa))
* Gallery embankments end flush with the gallery and its approach roads ([bf2d43c](https://github.com/Tachy/World-to-BeamNG/commit/bf2d43c59108d9326ed95fce362fa6aaba3e5ced))
* Generate the block stripe texture with a power-of-two height ([2c465e9](https://github.com/Tachy/World-to-BeamNG/commit/2c465e9b6d5bc70d29d40452ed6206c14c65ca85))
* Give block stripes the length and gap of the dashed divider ([bcd3fc0](https://github.com/Tachy/World-to-BeamNG/commit/bcd3fc024033c3f02f360204db1afec46be7e775))
* Give bridge nodes without an abutment the deck height ([4c20776](https://github.com/Tachy/World-to-BeamNG/commit/4c20776b9239143e6553f9adf0131842228e4d89))
* Give the block stripes the grey and opacity of the stock dashes ([42df7ad](https://github.com/Tachy/World-to-BeamNG/commit/42df7adb81ff3cd4046802a7a60d4f9d5a2d0d74))
* Give the road under a bridge a 45 degree embankment on both sides ([366f789](https://github.com/Tachy/World-to-BeamNG/commit/366f789df5c69a7d0643cd6a211e75aa5529d9a4))
* Interpolate the height of roads that pass under a bridge ([65e4e6d](https://github.com/Tachy/World-to-BeamNG/commit/65e4e6dfee46edffcc1e315cee7da39e455a9e8f))
* Keep each taper zone on its own half of a shared stretch ([ae1a332](https://github.com/Tachy/World-to-BeamNG/commit/ae1a332b858f7fcdb3fb322d061a5da4b9db7b13))
* Keep the main axis of a split on its OSM course ([edbf0ff](https://github.com/Tachy/World-to-BeamNG/commit/edbf0ff5a05ce790c9eefa7948a997f3115f15f3))
* Let split branches follow OSM once they reach their lane ([14e6767](https://github.com/Tachy/World-to-BeamNG/commit/14e67678a06d35406adbd13b563cff9590bfa894))
* Let the nearer road edge win where underpass embankments overlap ([c8caeb8](https://github.com/Tachy/World-to-BeamNG/commit/c8caeb8beafdf2249c8495670b8713440b02ec15))
* Let the terrain follow the blended road width along width transitions ([d3c0b5d](https://github.com/Tachy/World-to-BeamNG/commit/d3c0b5d9d1b9162466065cb9bae5403f172e9b1d))
* No grass and trees growing through bridge decks ([7b51560](https://github.com/Tachy/World-to-BeamNG/commit/7b51560c84c1818fe25375f7b123b9fb6e1eedbb))
* Skip tunnels that have no reachable portal in the map ([c7d985b](https://github.com/Tachy/World-to-BeamNG/commit/c7d985b0ff20be284ac7cb794102355d94e9b83e))
* Stop the block stripes before they paint over the double line ([4af62f2](https://github.com/Tachy/World-to-BeamNG/commit/4af62f2a8e817f19bb0fb289dc446dfb622a7b01))
* Take the underpass reference heights where the terrain height is stable ([e0fa80b](https://github.com/Tachy/World-to-BeamNG/commit/e0fa80bf436364b1993e6b1ad3fe4f26cf2329f9))
* Two lane changes within 100 m share the stretch between them ([3df5989](https://github.com/Tachy/World-to-BeamNG/commit/3df59893d8c5a2817f48694bc05cf8921a1ff5c1))


### Performance

* Build the bridge meshes once, in the export with the final widths ([075f7a0](https://github.com/Tachy/World-to-BeamNG/commit/075f7a0d559ec788d12dd47d6acd6613f44d6f69))


### Documentation

* Require English for all git/GitHub-facing text ([43b85c5](https://github.com/Tachy/World-to-BeamNG/commit/43b85c5db23f7f4e7105d2acfbe7fea0e5ff98fc))

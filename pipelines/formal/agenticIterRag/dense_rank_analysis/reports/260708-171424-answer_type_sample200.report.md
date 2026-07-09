# AIR endpoint 5100 样本 gold answer 类型抽样分析

## 基本信息
- created_at: `2026-07-08 17:14:24`
- manifest: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/manifest.json`
- dataset_jsonl: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/dataset.jsonl`
- full_sample_count: `5100`
- sampled_sample_count: `200`
- random_seed: `20260708`
- gold_answer_field: `reward_model.ground_truth.target`
- report_path: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/dense_rank_analysis/reports/260708-171424-answer_type_sample200.report.md`

## 结论先说
- 这 200 条里不只是 yes/no、年份、数字、短实体、常见词短语；还明显存在多答案列表、人名/标题实体、地点/国籍/语言、组织/作品/事件名、日期/事件时间表达、别名/修饰版本答案、较长描述短语。
- 对 reranker 训练最需要警惕的不是某一个类型，而是“短答案 + contains-any 证据规则”的组合。它会把只沾到一个词、一个数字、一个年份的 passage 当成正例。
- 多答案列表也要单独看。一个 doc 只包含列表中的一个 answer，也会被 contains-any 判成 hit；如果问题本来要求多个答案，这种正例偏宽。
- 人名和较完整实体通常比数字/年份稳定，但仍然不是纯净 gold doc。演员表、列表页、消歧义页都可能只提到名字，不真正回答当前问题。

## 样本级主类型分布

| type | count | pct |
| --- | ---: | ---: |
| `person_name_or_title_entity` | 65 | 32.5% |
| `date_or_time_expression` | 33 | 16.5% |
| `short_entity_or_short_phrase` | 26 | 13.0% |
| `year` | 15 | 7.5% |
| `common_noun_phrase` | 14 | 7.0% |
| `yes_no` | 12 | 6.0% |
| `multi_answer_list` | 8 | 4.0% |
| `multi_answer_set_with_aliases` | 7 | 3.5% |
| `geo_or_nationality` | 7 | 3.5% |
| `entity_or_phrase_other` | 4 | 2.0% |
| `number_or_ordinal` | 4 | 2.0% |
| `long_descriptive_phrase` | 2 | 1.0% |
| `acronym_or_code` | 2 | 1.0% |
| `organization_work_or_event` | 1 | 0.5% |

说明：样本级会优先识别多答案样本，所以一个包含多个演员名的样本会进入 `multi_answer_list`，不会被简单算成人名。

## 单个 answer span 主类型分布

| type | count | pct |
| --- | ---: | ---: |
| `person_name_or_title_entity` | 92 | 39.7% |
| `short_entity_or_short_phrase` | 35 | 15.1% |
| `date_or_time_expression` | 33 | 14.2% |
| `common_noun_phrase` | 19 | 8.2% |
| `year` | 15 | 6.5% |
| `yes_no` | 12 | 5.2% |
| `geo_or_nationality` | 9 | 3.9% |
| `long_descriptive_phrase` | 4 | 1.7% |
| `entity_or_phrase_other` | 4 | 1.7% |
| `acronym_or_code` | 4 | 1.7% |
| `number_or_ordinal` | 4 | 1.7% |
| `organization_work_or_event` | 1 | 0.4% |

说明：span 级是把每个 gold answer 字符串单独拆开看，所以同一个样本有 3 个 gold answers，就会贡献 3 个 span。

## 风险标签分布

| type | count | pct |
| --- | ---: | ---: |
| `short_answer` | 187 | 80.6% |
| `proper_name` | 92 | 39.7% |
| `contains_number` | 58 | 25.0% |
| `short_entity` | 48 | 20.7% |
| `time_like` | 33 | 14.2% |
| `common_phrase` | 19 | 8.2% |
| `leading_fragment` | 15 | 6.5% |
| `year_like` | 15 | 6.5% |
| `boolean_answer` | 12 | 5.2% |
| `longer_phrase` | 10 | 4.3% |
| `numeric_or_ordinal` | 4 | 1.7% |

说明：风险标签是可重叠的。比如 `French` 既是 `short_answer`，也可能是 `short_entity`；`In 1642` 既是年份，也带有 `leading_fragment`。

## 类型解释与例子

### `common_noun_phrase`

这类是普通名词短语或职业身份，比如 film director、professional tennis player。它们不是唯一实体，很多人物页面都会出现，所以字符串命中通常只能说明类型对，不一定说明答案对。

例子：
- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `one of America's most influential evangelicals` | sample_id=`sample-000228:step:2` | question: KCFO (970 AM) airs the show of an American historical theologian that has been described as what? | sub_query: KCFO 970 AM show description
- `within the inner membrane` | sample_id=`sample-000490:step:1` | question: where is the matrix located in the mitochondria? | sub_query: where is the matrix located in the mitochondria
- `the Treaty on the Functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `the Phantom` | sample_id=`sample-001055:step:2` | question: who does christine choose in love never dies? | sub_query: who does christine choose in love never dies musical
- `the Caribbean` | sample_id=`sample-001178:step:1` | question: where is saba university school of medicine located? | sub_query: Saba University School of Medicine location
- `sextuple overtime` | sample_id=`sample-001640:step:2` | question: record number of overtime periods in nhl playoffs? | sub_query: record number of overtime periods in nhl playoffs
- `less than a mile to the east` | sample_id=`sample-001855:step:4` | question: How close is Wrigley Field to the lake supplying drinking water to the birth city of The Adventurer's performer? | sub_query: lake supplying drinking water to birth city of The Adventurer's performer

### `multi_answer_set_with_aliases`

这类样本有多个答案字符串，其中一些是同一个答案的别名或带修饰版本。它有利于召回，但也可能让统计时看起来像多答案。

例子：
- `Washington County | Washington County, Maryland` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `the Treaty on the Functioning of the European Union | Treaty on the Functioning of the European Union | Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `less than a mile to the east | Mile` | sample_id=`sample-001855:step:4` | question: How close is Wrigley Field to the lake supplying drinking water to the birth city of The Adventurer's performer? | sub_query: lake supplying drinking water to birth city of The Adventurer's performer
- `Wembley Stadium | Wembley` | sample_id=`sample-002230:step:3` | question: Where did the torch relay begin in the place of birth of On Her Majesty's Secret Service's author for the 2008 Olympics? | sub_query: where did the torch relay begin for 2008 olympics
- `The Rolling Stones | Rolling Stones` | sample_id=`sample-003447:step:1` | question: Depp based his pirate character on the guitarist of what band? | sub_query: Depp based his pirate character on the guitarist of what band
- `America | the US | U.S. | the United States | United States | US` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `control of more Luftwaffe units | Luftwaffe` | sample_id=`sample-004512:step:2` | question: What did Goring believe the operator of battleships from the country of the military that follows the Reichswehr woul... | sub_query: What did Hermann Goring believe the operator of battleships from the country of the military that...

### `date_or_time_expression`

这类是更完整的时间表达，比如月份日期、9/11、世纪、年代。它比纯年份更具体一些，但如果问题问的是事件关系，单纯出现这个时间仍然不一定是答案证据。

例子：
- `February 17, 2016` | sample_id=`sample-000119:step:1` | question: when did i hate u i love u come out? | sub_query: I Hate U I Love U release date
- `12/8 time` | sample_id=`sample-000331:step:2` | question: What time signature is the first track on Phil Spector's "Back to Mono" in? | sub_query: To Know Him Is to Love Him time signature
- `August 3 - 12, 2018` | sample_id=`sample-000374:step:1` | question: When is the Sturgis Motorcycle Rally in the state The Jumping-Off Place takes place in? | sub_query: The Jumping-Off Place South Dakota
- `29 December 2017` | sample_id=`sample-000394:step:2` | question: when will film stars don't die in liverpool be released? | sub_query: Film Stars Don't Die in Liverpool release date US
- `22 September 1878` | sample_id=`sample-000463:step:1` | question: What is the founding date of this largest and oldest independent theatrical arts school in Russia, where Maximilian M... | sub_query: founding date of Russian Academy of Theatre Arts
- `August 2 -- 5` | sample_id=`sample-000560:step:3` | question: When is the golf tournament in Akron, in the state Hotel Courtland is located? | sub_query: WGC-Bridgestone Invitational date
- `November 20, 1969` | sample_id=`sample-000837:step:2` | question: When did the lottery start in the state where Holly Brooks was educated? | sub_query: when was washington lottery established
- `March 30, 2018` | sample_id=`sample-000940:step:1` | question: when does a series of unfortunate events season two come out on netflix? | sub_query: A Series of Unfortunate Events Season 2 release date Netflix

### `short_entity_or_short_phrase`

这类是短实体或短短语，不一定能判断是人、地、组织还是概念。短答案最大的问题是上下文约束弱，字符串命中不能代表语义命中。

例子：
- `Istanbul` | sample_id=`sample-000140:step:2` | question: Where is the Blue Mosque located, in the country where the village of Belen is found? | sub_query: Blue Mosque location
- `opera` | sample_id=`sample-000139:step:3` | question: What was the genre of the Sextet composer? | sub_query: minimal music genre
- `Opera` | sample_id=`sample-000139:step:3` | question: What was the genre of the Sextet composer? | sub_query: minimal music genre
- `Katherina` | sample_id=`sample-000277:step:1` | question: who was the shrew in taming ofthe shrew? | sub_query: who is the shrew in the taming of the shrew
- `Bristol, England` | sample_id=`sample-000488:step:1` | question: Robert Maynard was famous for his defeat of the pirate that was probably born where? | sub_query: Blackbeard birthplace
- `Hollywood` | sample_id=`sample-000486:step:2` | question: Where did the director of film Confession (1937 Film) die? | sub_query: Joe May death place
- `Zaire` | sample_id=`sample-000734:step:0` | question: what was the former name of the democratic republic of congo? | sub_query: former name of the Democratic Republic of the Congo
- `38 million` | sample_id=`sample-001138:step:2` | question: How many records was sold by the performer of Super Collider worldwide? | sub_query: worldwide sales of Megadeth Super Collider

### `person_name_or_title_entity`

这类大多是人名或首字母大写的标题实体，比如 Ron Howard。完整人名通常是比较好的答案锚点，但同名、列表页、演员表也会带来弱噪声。

例子：
- `Washington County` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `Washington County, Maryland` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `Bob Marley` | sample_id=`sample-000163:step:1` | question: who made the song i shot the sheriff? | sub_query: I Shot the Sheriff song writer
- `E1 Entertainment` | sample_id=`sample-000192:step:1` | question: Which entertainment group released a mixtape which was made by a member of the Lox? | sub_query: Lox member mixtape entertainment group
- `San Andreas` | sample_id=`sample-000197:step:2` | question: What is the capital of the county where Mountain Ranch is located? | sub_query: capital of Calaveras County
- `Forbidden Priests` | sample_id=`sample-000309:step:2` | question: Which film has the director who died earlier, Forbidden Priests or The Clown (1976 Film)? | sub_query: Forbidden Priests director death date
- `Sony Pictures Studios` | sample_id=`sample-000319:step:1` | question: where is the tv show the goldbergs filmed? | sub_query: where is the tv show the goldbergs filmed
- `Draymond Green` | sample_id=`sample-000578:step:2` | question: who led the nba in steals this year? | sub_query: nba steals leader 2023

### `geo_or_nationality`

这类是地点、国家、州名、民族或语言，比如 Kentucky、French。单词越短越容易歧义，French 既可能是语言、国籍，也可能只是形容词。

例子：
- `France` | sample_id=`sample-000231:step:1` | question: which country gave the united states the statue of liberty as a gift? | sub_query: which country gave the united states the statue of liberty as a gift
- `American` | sample_id=`sample-000473:step:2` | question: Which country the director of film The Unchastened Woman (1918 Film) is from? | sub_query: Edward José director of The Unchastened Woman country
- `British` | sample_id=`sample-000553:step:1` | question: The physicist who won the 1917 Nobel Prize in Physics for the discovery of Characteristic X Rays was of what national... | sub_query: Charles Glover Barkla nationality
- `India` | sample_id=`sample-000922:step:1` | question: Which country the director of film Money Hai Toh Honey Hai is from? | sub_query: Ganesh Acharya country
- `America` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `United States` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `California` | sample_id=`sample-004306:step:1` | question: what state does the fosters take place in? | sub_query: Fosters (TV series) location
- `New York` | sample_id=`sample-004565:step:2` | question: Where was the place of death of the performer of song Sophisticated Lady? | sub_query: Duke Ellington death location

### `long_descriptive_phrase`

这类是较长描述短语。它的误命中率一般低一些，但 exact phrase 匹配会更脆，换一种说法就可能漏掉。

例子：
- `Salt Lake City metropolitan area` | sample_id=`sample-000270:step:2` | question: Which part of Utah contains the city where Charles Halford was born? | sub_query: Charles Halford birthplace
- `Treaty on the Functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `Many of majority European ancestry and appearance "married white" and assimilated into white society` | sample_id=`sample-003558:step:2` | question: How did many multiracial individuals, of the nation that started making an unlicensed version of the 40mm at the begi... | sub_query: how multiracial individuals in Germany attained social and economic advantages after WWII

### `entity_or_phrase_other`

这类是启发式规则没法稳定归到上面类型的答案。它不一定有问题，只是需要人工看上下文后再决定是否适合字符串证据规则。

例子：
- `Days of the New` | sample_id=`sample-000801:step:2` | question: Which band was formed first, Days of the New or The Accidentals? | sub_query: Days of the New formation date The Accidentals formation date
- `British playwright Alan Ayckbourn` | sample_id=`sample-001804:step:1` | question: who has written a snake in the grass? | sub_query: author of snake in the grass play
- `350 to 450` | sample_id=`sample-002673:step:1` | question: how many types of french cheese are there? | sub_query: number of types of french cheese
- `The British Invasion` | sample_id=`sample-005086:step:1` | question: when does rita and dexter get back together? | sub_query: when does rita and dexter get back together

### `yes_no`

这类答案只有 yes/no。它对字符串命中奖励最危险，因为 yes/no 在普通段落里太常见；即便边界匹配也只是减少 Yeshiva 这种子串误伤，不能保证语义正确。

例子：
- `no` | sample_id=`sample-000992:step:2` | question: Are Cowles Mead and Mickaël Mazzoli both from the same country? | sub_query: Mickaël Mazzoli country
- `no` | sample_id=`sample-001206:step:1` | question: Are Pylon and Team Sleep experimental bands? | sub_query: Is Pylon an experimental band
- `no` | sample_id=`sample-001624:step:3` | question: Do both films, Captive (2015 film) and Arsène Lupin Returns, have the directors who are from the same country? | sub_query: Jean-Paul Salomé country
- `no` | sample_id=`sample-002239:step:2` | question: Are Pizza Ranch and Gino's East located in Florida? | sub_query: Are Pizza Ranch and Gino's East located in Florida
- `no` | sample_id=`sample-002446:step:2` | question: Are both Perranuthnoe and Rostami, Hormozgan located in the same country? | sub_query: Rostami, Hormozgan country
- `no` | sample_id=`sample-002896:step:2` | question: Are Vallea and Oroya both genii of cacti? | sub_query: Vallea and Oroya genii of cacti
- `yes` | sample_id=`sample-003652:step:1` | question: Do both directors of films They Shoot Horses, Don'T They? (Film) and Blackboard Jungle share the same nationality? | sub_query: nationality of Sydney Pollack and Richard Brooks
- `no` | sample_id=`sample-003870:step:2` | question: Are Give Us Our Skeletons and Dave Chappelle's Block Party both comedies? | sub_query: Is Dave Chappelle's Block Party a comedy

### `multi_answer_list`

这类样本有多个不同 gold answer，通常是列表题或多跳结果。contains-any 规则会把只包含其中一个答案的 doc 当正例，这对训练 reranker 可能偏宽。

例子：
- `Saba | the Caribbean` | sample_id=`sample-001178:step:1` | question: where is saba university school of medicine located? | sub_query: Saba University School of Medicine location
- `Lisandra Tena | Dayton Callie | Daniel Sharman | Sam Underwood` | sample_id=`sample-001174:step:1` | question: fear the walking dead season 3 new characters? | sub_query: Fear the Walking Dead season 3 new characters list
- `Samantha Hitchcock | Tamati Coffey` | sample_id=`sample-003498:step:1` | question: who won dancing with the stars nz 2009? | sub_query: Dancing with the Stars NZ 2009 winner
- `Jason Segel | Cobie Smulders | Neil Patrick Harris | Josh Radnor | Alyson Hannigan | Cristin Milioti` | sample_id=`sample-003493:step:1` | question: who are the stars of how i met your mother? | sub_query: main stars of how i met your mother
- `Paul Warfield Tibbets Jr. | Paul Tibbets` | sample_id=`sample-003712:step:1` | question: Who was the pilot who dropped the atomic bomb on the first Japanese city nuked by the United States? | sub_query: Who was the pilot who dropped the atomic bomb on the first Japanese city nuked by the United States?
- `Al Unser | A. J. Foyt | Rick Mears` | sample_id=`sample-003866:step:1` | question: who has won the indy 500 the most times? | sub_query: most indy 500 wins
- `Tom Petty | Reverend Ike | Reba McEntire | Willie Nelson` | sample_id=`sample-003909:step:1` | question: who sings mind your own business with hank jr? | sub_query: who sings mind your own business with hank jr
- `Colombia | Argentina | Uruguay | Brazil` | sample_id=`sample-004045:step:3` | question: teams qualified for world cup 2018 in south america? | sub_query: other two south american teams that qualified for 2018 world cup

### `acronym_or_code`

这类是缩写或代码。短缩写如果很独特还好，但像 US、UK、TV 这种会非常泛，必须结合问题实体看。

例子：
- `SBS` | sample_id=`sample-001400:step:2` | question: On which station could you have watched the actor from "My Fair Lady" play a character in "I Can Hear Your Voice"? | sub_query: I Can Hear Your Voice station
- `U.S.` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `US` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `F5` | sample_id=`sample-004882:step:1` | question: what was the worst tornado to hit the united states? | sub_query: worst tornado to hit the united states

### `number_or_ordinal`

这类是数字、数量或序数，比如 7、eighth。它通常比年份还脆，因为数字本身缺少实体约束，很多无关 passage 都可能包含同一个数字。

例子：
- `22` | sample_id=`sample-001547:step:2` | question: how many episodes in person of interest season 2? | sub_query: how many episodes in person of interest season 2
- `34th` | sample_id=`sample-004841:step:2` | question: What is the population ranking of the state that is the narrative location of Tishomingo Blues? | sub_query: population ranking of Mississippi
- `53` | sample_id=`sample-004884:step:1` | question: how many house of representatives are there in california? | sub_query: how many house of representatives are there in california
- `24` | sample_id=`sample-005065:step:2` | question: fairy tail how many episodes in season 2? | sub_query: Fairy Tail season 2 episode count

### `year`

这类答案主要是年份，比如 1974 或 In 1642。年份在百科文本里出现频率很高，命中年份不等于命中问题证据，容易把背景时间误当正例。

例子：
- `1611` | sample_id=`sample-002339:step:1` | question: when did the king james bible come out? | sub_query: when was the king james bible first published
- `1853` | sample_id=`sample-002381:step:1` | question: when did the last convict ship reach tasmania? | sub_query: last convict ship to reach tasmania
- `1791` | sample_id=`sample-002406:step:1` | question: when did the latin american independence movement began? | sub_query: start of latin american independence movement
- `1917` | sample_id=`sample-002595:step:2` | question: In what year was Yuk Young-soo's husband born? | sub_query: Park Chung-hee birth year
- `1959` | sample_id=`sample-003935:step:1` | question: when was the linq hotel in las vegas built? | sub_query: when was the linq hotel in las vegas built
- `1827` | sample_id=`sample-004052:step:1` | question: In what year was slavery eliminated in the state where Raymond from Everybody Loves Raymond lives? | sub_query: year slavery abolished in Illinois
- `2013` | sample_id=`sample-004181:step:1` | question: when's the last time tiger woods won a tournament? | sub_query: last tournament win tiger woods
- `By 1975` | sample_id=`sample-004268:step:1` | question: when did it become legal to own gold again? | sub_query: when did it become legal to own gold again

### `organization_work_or_event`

这类是组织、作品、赛事、事件等名字。通常比纯数字稳定，但如果名字里有通用词，比如 attack、party、college，也可能命中背景介绍而不是答案关系。

例子：
- `song` | sample_id=`sample-003760:step:0` | question: what does Everybody Get Up and I Love Rock 'n' Roll have in common? | sub_query: Everybody Get Up and I Love Rock 'n' Roll

## 重点风险标签例子

### `short_answer`

- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `Washington County` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `Washington County | Washington County, Maryland` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `Istanbul` | sample_id=`sample-000140:step:2` | question: Where is the Blue Mosque located, in the country where the village of Belen is found? | sub_query: Blue Mosque location
- `Istanbul` | sample_id=`sample-000140:step:2` | question: Where is the Blue Mosque located, in the country where the village of Belen is found? | sub_query: Blue Mosque location
- `opera` | sample_id=`sample-000139:step:3` | question: What was the genre of the Sextet composer? | sub_query: minimal music genre
- `Opera` | sample_id=`sample-000139:step:3` | question: What was the genre of the Sextet composer? | sub_query: minimal music genre

### `contains_number`

- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `February 17, 2016` | sample_id=`sample-000119:step:1` | question: when did i hate u i love u come out? | sub_query: I Hate U I Love U release date
- `February 17, 2016` | sample_id=`sample-000119:step:1` | question: when did i hate u i love u come out? | sub_query: I Hate U I Love U release date
- `E1 Entertainment` | sample_id=`sample-000192:step:1` | question: Which entertainment group released a mixtape which was made by a member of the Lox? | sub_query: Lox member mixtape entertainment group
- `E1 Entertainment` | sample_id=`sample-000192:step:1` | question: Which entertainment group released a mixtape which was made by a member of the Lox? | sub_query: Lox member mixtape entertainment group
- `12/8 time` | sample_id=`sample-000331:step:2` | question: What time signature is the first track on Phil Spector's "Back to Mono" in? | sub_query: To Know Him Is to Love Him time signature
- `12/8 time` | sample_id=`sample-000331:step:2` | question: What time signature is the first track on Phil Spector's "Back to Mono" in? | sub_query: To Know Him Is to Love Him time signature

### `common_phrase`

- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `since 2010` | sample_id=`sample-000002:step:3` | question: How long has the product Fog's End is known for producing been legal in the United States? | sub_query: When did Fog's End Distillery become legal in the United States
- `one of America's most influential evangelicals` | sample_id=`sample-000228:step:2` | question: KCFO (970 AM) airs the show of an American historical theologian that has been described as what? | sub_query: KCFO 970 AM show description
- `one of America's most influential evangelicals` | sample_id=`sample-000228:step:2` | question: KCFO (970 AM) airs the show of an American historical theologian that has been described as what? | sub_query: KCFO 970 AM show description
- `within the inner membrane` | sample_id=`sample-000490:step:1` | question: where is the matrix located in the mitochondria? | sub_query: where is the matrix located in the mitochondria
- `within the inner membrane` | sample_id=`sample-000490:step:1` | question: where is the matrix located in the mitochondria? | sub_query: where is the matrix located in the mitochondria
- `the Treaty on the Functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `the Treaty on the Functioning of the European Union | Treaty on the Functioning of the European Union | Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...

### `leading_fragment`

- `the Treaty on the Functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `the Treaty on the Functioning of the European Union | Treaty on the Functioning of the European Union | Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `the Phantom` | sample_id=`sample-001055:step:2` | question: who does christine choose in love never dies? | sub_query: who does christine choose in love never dies musical
- `the Phantom` | sample_id=`sample-001055:step:2` | question: who does christine choose in love never dies? | sub_query: who does christine choose in love never dies musical
- `the Caribbean` | sample_id=`sample-001178:step:1` | question: where is saba university school of medicine located? | sub_query: Saba University School of Medicine location
- `Saba | the Caribbean` | sample_id=`sample-001178:step:1` | question: where is saba university school of medicine located? | sub_query: Saba University School of Medicine location
- `The Memory Will Never Die` | sample_id=`sample-001674:step:2` | question: What song by One Thing Remains was used by World Wrestling Entertainment for their event on April 1, 2007, at Ford Fi... | sub_query: One Thing Remains song used at WrestleMania 23
- `The Memory Will Never Die` | sample_id=`sample-001674:step:2` | question: What song by One Thing Remains was used by World Wrestling Entertainment for their event on April 1, 2007, at Ford Fi... | sub_query: One Thing Remains song used at WrestleMania 23

### `multi_target`

- `Washington County | Washington County, Maryland` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `the Treaty on the Functioning of the European Union | Treaty on the Functioning of the European Union | Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `Saba | the Caribbean` | sample_id=`sample-001178:step:1` | question: where is saba university school of medicine located? | sub_query: Saba University School of Medicine location
- `Lisandra Tena | Dayton Callie | Daniel Sharman | Sam Underwood` | sample_id=`sample-001174:step:1` | question: fear the walking dead season 3 new characters? | sub_query: Fear the Walking Dead season 3 new characters list
- `less than a mile to the east | Mile` | sample_id=`sample-001855:step:4` | question: How close is Wrigley Field to the lake supplying drinking water to the birth city of The Adventurer's performer? | sub_query: lake supplying drinking water to birth city of The Adventurer's performer
- `Wembley Stadium | Wembley` | sample_id=`sample-002230:step:3` | question: Where did the torch relay begin in the place of birth of On Her Majesty's Secret Service's author for the 2008 Olympics? | sub_query: where did the torch relay begin for 2008 olympics
- `The Rolling Stones | Rolling Stones` | sample_id=`sample-003447:step:1` | question: Depp based his pirate character on the guitarist of what band? | sub_query: Depp based his pirate character on the guitarist of what band
- `Samantha Hitchcock | Tamati Coffey` | sample_id=`sample-003498:step:1` | question: who won dancing with the stars nz 2009? | sub_query: Dancing with the Stars NZ 2009 winner

### `contains_alias_variant`

- `Washington County | Washington County, Maryland` | sample_id=`sample-000067:step:2` | question: In which county is the city to which WHGT licensed to broadcast? | sub_query: Washington County, Maryland
- `the Treaty on the Functioning of the European Union | Treaty on the Functioning of the European Union | Treaty on the functioning of the European Union` | sample_id=`sample-000601:step:1` | question: What treaty formed the basis of the union of nations who made a statement about the chemical used in warfare? | sub_query: treaty formed the basis of the union of nations who made a statement about the chemical used in w...
- `less than a mile to the east | Mile` | sample_id=`sample-001855:step:4` | question: How close is Wrigley Field to the lake supplying drinking water to the birth city of The Adventurer's performer? | sub_query: lake supplying drinking water to birth city of The Adventurer's performer
- `Wembley Stadium | Wembley` | sample_id=`sample-002230:step:3` | question: Where did the torch relay begin in the place of birth of On Her Majesty's Secret Service's author for the 2008 Olympics? | sub_query: where did the torch relay begin for 2008 olympics
- `The Rolling Stones | Rolling Stones` | sample_id=`sample-003447:step:1` | question: Depp based his pirate character on the guitarist of what band? | sub_query: Depp based his pirate character on the guitarist of what band
- `America | the US | U.S. | the United States | United States | US` | sample_id=`sample-004123:step:2` | question: What is the country of citizenship of the person who created the publisher of On the Internet: nobody knows you're a ... | sub_query: country of citizenship of Peter Steiner
- `control of more Luftwaffe units | Luftwaffe` | sample_id=`sample-004512:step:2` | question: What did Goring believe the operator of battleships from the country of the military that follows the Reichswehr woul... | sub_query: What did Hermann Goring believe the operator of battleships from the country of the military that...

## 口语化解读

如果我们继续用 answer string 去找 true answer doc，最危险的是那种“看起来命中了，其实只是碰巧出现”的情况。yes/no 是最明显的，年份和数字也类似；它们本身信息量太低，不能证明这个 doc 真的回答了问题。

短实体也要小心。像一个州名、一个语言名、一个姓氏，命中以后只能说明 passage 提到了这个词，不能说明它处在正确关系里。对于 reranker 来说，这会把一些语义不够正的文档推成正例，训练信号会变软。

多答案样本是另一个问题。比如问题要求三个人或三个分支，gold answer 里有多个字符串。当前 contains-any 逻辑只要命中其中一个就算 evidence hit，这对召回分析是宽松的，但对训练 reranker 未必理想，因为 reranker 学到的可能是“包含任意一个答案片段就够了”。

比较可靠的类型一般是完整人名、独特组织名、独特作品名、较长事件名。但这里也不能完全放心，因为 Wikipedia passage 可能只是列表式提到实体，没有提供问题需要的关系证据。

所以后续如果要清洗 hard subset，我建议不要只做 yes/no 排除。更稳的路线是分层：先排除 yes/no；再对年份、纯数字、极短 answer 单独降权或要求 query/entity 共现；最后对多答案样本考虑 all/partial hit 的区别，而不是简单 contains-any。

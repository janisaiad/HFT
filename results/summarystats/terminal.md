
(HFT) janis@pc-samsung-janis:/c/Users/janis/AppData/Local/Programs/cursor$ /home/janis/HFTP2/HFT/.venv/bin/python /home/janis/HFTP2/HFT/models/summary/summarystatistics.py
Calcul de la taille des dossiers...
Nombre total de stocks trouvés: 46
Top 5 plus gros dossiers:
GOOGL: 5.51 GB
AAPL: 3.08 GB
AMZN: 3.06 GB
AAL: 2.13 GB
MSFT: 2.04 GB
Processing stocks:   0%|                            | 0/46 [00:00<?, ?it/sDebug: Tick size = 2.842170943040401e-14                                    
Debug: Min price = 147.525
Debug: Max price = 183.34
Debug: Average spread = 666702115338.0367
Debug: Trades at bid = 1579770 / 2828890
Debug: Jumps count = 467
Processing stocks:   2%|▍                   | 1/46 [01:00<45:19, 60.43s/itDebug: Tick size = 2.842170943040401e-14                                    
Debug: Min price = 219.70999999999998
Debug: Max price = 250.2
Debug: Average spread = 871622611716.0636
Debug: Trades at bid = 1352797 / 2456867
Debug: Jumps count = 483
Processing stocks:   4%|▊                   | 2/46 [01:36<33:49, 46.13s/it^Processing stocks:   4%|▊                   | 2/46 [01:46<39:11, 53.44s/it]
Traceback (most recent call last):                                         
  File "/home/janis/HFTP2/HFT/models/summary/summarystatistics.py", line 322, in <module>
    stats_accumulator.update(df)
  File "/home/janis/HFTP2/HFT/models/summary/summarystatistics.py", line 189, in update
    if abs(diff) >= self.tick_size:  # Saut d'au moins un tick
       ^^^^^^^^^
KeyboardInterrupt

(HFT) janis@pc-samsung-janis:/c/Users/janis/AppData/Local/Programs/cursor$ pkill -9 python
(HFT) janis@pc-samsung-janis:/c/Users/janis/AppData/Local/Programs/cursor$ /home/janis/HFTP2/HFT/.venv/bin/python /home/janis/HFTP2/HFT/models/summary/summarystatistics.py
Suppression: results/summarystats/AAPL_stats.json
Suppression: results/summarystats/GOOGL_stats.json
Calcul de la taille des dossiers...
Nombre total de stocks trouvés: 46
Top 5 plus gros dossiers:
GOOGL: 5.51 GB
AAPL: 3.08 GB
AMZN: 3.06 GB
AAL: 2.13 GB
MSFT: 2.04 GB
Processing stocks:   0%|                            | 0/46 [00:00<?, ?it/sStatistiques pour le tick: 0.01                                             
Min price: 147.525
Max price: 183.34
Average spread: 1.894881379877339
Trades at bid: 55.844165025858196%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GOOGL_stats.json
Processing stocks:   2%|▎                | 1/46 [01:45<1:19:23, 105.85s/itStatistiques pour le tick: 0.01                                             
Min price: 219.70999999999998
Max price: 250.2
Average spread: 2.4773004603163824
Trades at bid: 55.06187351614882%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/AAPL_stats.json
Processing stocks:   4%|▊                   | 2/46 [02:30<51:01, 69.58s/itStatistiques pour le tick: 0.01                                             
Min price: 180.565
Max price: 232.065
Average spread: 2.963819890054184
Trades at bid: 50.478430058291934%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/AMZN_stats.json
Processing stocks:   7%|█▎                  | 3/46 [03:00<37:08, 51.83s/itStatistiques pour le tick: 0.01                                             
Min price: 8.945
Max price: 19.19
Average spread: 1.3065243781844242
Trades at bid: 59.97836243952074%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/AAL_stats.json
Processing stocks:   9%|█▋                  | 4/46 [03:41<33:15, 47.51s/itStatistiques pour le tick: 0.01                                             
Min price: 405.46500000000003
Max price: 455.755
Average spread: 8.78373335668109
Trades at bid: 53.40771591354704%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/MSFT_stats.json
Processing stocks:  11%|██▏                 | 5/46 [03:59<25:14, 36.95s/itStatistiques pour le tick: 0.01                                             
Min price: 7.3149999999999995
Max price: 20.435000000000002
Average spread: 1.5922521051008731
Trades at bid: 59.79204920660076%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GT_stats.json
Processing stocks:  13%|██▌                 | 6/46 [04:28<22:46, 34.16s/itStatistiques pour le tick: 0.01                                             
Min price: 18.520000000000003
Max price: 33.39
Average spread: 1.1595543386567446
Trades at bid: 59.98455542898255%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/INTC_stats.json
Processing stocks:  15%|███                 | 7/46 [04:38<17:07, 26.36s/itStatistiques pour le tick: 0.01                                             
Min price: 6.32
Max price: 20.165
Average spread: 1.584548406226785
Trades at bid: 59.42484589937752%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/PTEN_stats.json
Processing stocks:  17%|███▍                | 8/46 [04:55<14:43, 23.24s/itStatistiques pour le tick: 0.01                                             
Min price: 3.145
Max price: 51.355
Average spread: 1.655964498595585
Trades at bid: 58.62888945472008%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/IOVA_stats.json
Processing stocks:  20%|███▉                | 9/46 [05:10<12:43, 20.63s/itStatistiques pour le tick: 0.01                                             
Min price: 4.465
Max price: 16.985
Average spread: 1.6917714662822305
Trades at bid: 61.59034008007421%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/MLCO_stats.json
Processing stocks:  22%|████▏              | 10/46 [05:30<12:12, 20.34s/itStatistiques pour le tick: 0.01                                             
Min price: 2.705
Max price: 10.585
Average spread: 1.2365789875740254
Trades at bid: 61.38783434143848%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/PTON_stats.json
Processing stocks:  24%|████▌              | 11/46 [05:45<10:57, 18.80s/itStatistiques pour le tick: 0.01                                             
Min price: 4.29
Max price: 14.48
Average spread: 1.5099988867382264
Trades at bid: 59.93377483443708%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/VLY_stats.json
Processing stocks:  26%|████▉              | 12/46 [06:02<10:20, 18.25s/itStatistiques pour le tick: 0.01                                             
Min price: 8.024999999999999
Max price: 10.355
Average spread: 1.1319330718421352
Trades at bid: 62.274034961081334%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/VOD_stats.json
Processing stocks:  28%|█████▎             | 13/46 [06:18<09:42, 17.65s/itStatistiques pour le tick: 0.01                                             
Min price: 2.465
Max price: 11.17
Average spread: 2.5760171453300265
Trades at bid: 63.400416277383584%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/AMRX_stats.json
Processing stocks:  30%|█████▊             | 14/46 [06:37<09:38, 18.08s/itStatistiques pour le tick: 0.01                                             
Min price: 5.34
Max price: 16.175
Average spread: 2.66931085044585
Trades at bid: 57.99736473552055%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/AVDX_stats.json
Processing stocks:  33%|██████▏            | 15/46 [06:47<08:00, 15.49s/itStatistiques pour le tick: 0.01                                             
Min price: 31.905
Max price: 37.18
Average spread: 1.728285523215231
Trades at bid: 57.49005943619392%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/CSX_stats.json
Processing stocks:  35%|██████▌            | 16/46 [06:53<06:24, 12.82s/itStatistiques pour le tick: 0.01                                             
Min price: 7.035
Max price: 15.785
Average spread: 1.6865223383471117
Trades at bid: 60.91285352114733%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/WB_stats.json
Processing stocks:  37%|███████            | 17/46 [07:04<05:52, 12.16s/itStatistiques pour le tick: 0.01                                             
Min price: 3.295
Max price: 12.8
Average spread: 2.2553573194572785
Trades at bid: 62.101252448951584%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/BGC_stats.json
Processing stocks:  39%|███████▍           | 18/46 [07:14<05:20, 11.45s/itStatistiques pour le tick: 0.01                                             
Min price: 2.9050000000000002
Max price: 5.695
Average spread: 1.0956699226201907
Trades at bid: 71.05438102653171%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GRAB_stats.json
Processing stocks:  41%|███████▊           | 19/46 [07:24<04:55, 10.95s/itStatistiques pour le tick: 0.01                                             
Min price: 32.565
Max price: 36.81
Average spread: 1.6044829105026366
Trades at bid: 58.94758953891398%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/KHC_stats.json
Processing stocks:  43%|████████▎          | 20/46 [07:28<03:50,  8.86s/itStatistiques pour le tick: 0.01                                             
Min price: 5.13
Max price: 15.615
Average spread: 5.0225610439431465
Trades at bid: 59.847135705277644%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/HLMN_stats.json
Processing stocks:  46%|████████▋          | 21/46 [07:34<03:27,  8.30s/itStatistiques pour le tick: 0.01                                             
Min price: 9.715
Max price: 44.93
Average spread: 8.560593898541123
Trades at bid: 58.46010165412453%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/IEP_stats.json
Processing stocks:  48%|█████████          | 22/46 [07:43<03:20,  8.36s/itStatistiques pour le tick: 0.01                                             
Min price: 9.905
Max price: 19.355
Average spread: 2.3798496898299297
Trades at bid: 63.62536114839502%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GBDC_stats.json
Processing stocks:  50%|█████████▌         | 23/46 [07:51<03:06,  8.13s/itStatistiques pour le tick: 0.01                                             
Min price: 6.6
Max price: 8.85
Average spread: 1.1556872723102278
Trades at bid: 63.321374333676474%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/WBD_stats.json
Processing stocks:  52%|█████████▉         | 24/46 [07:54<02:26,  6.65s/itStatistiques pour le tick: 0.01                                             
Min price: 0.613
Max price: 4.855
Average spread: 1.0036503553844738
Trades at bid: 60.06071091928362%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/PSNY_stats.json
Processing stocks:  54%|██████████▎        | 25/46 [08:03<02:33,  7.33s/itStatistiques pour le tick: 0.01                                             
Min price: 107.27000000000001
Max price: 161.95
Average spread: 23.69603589343797
Trades at bid: 53.181234674189604%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/NTAP_stats.json
Processing stocks:  57%|██████████▋        | 26/46 [08:05<01:54,  5.73s/itStatistiques pour le tick: 0.01                                             
Min price: 10.76
Max price: 29.939999999999998
Average spread: 3.8627823470631504
Trades at bid: 53.880022014309304%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GEO_stats.json
Processing stocks:  59%|███████████▏       | 27/46 [08:07<01:29,  4.73s/itStatistiques pour le tick: 0.01                                             
Min price: 2.525
Max price: 4.425
Average spread: 1.1002397707346192
Trades at bid: 65.58115658749432%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/LCID_stats.json
Processing stocks:  61%|███████████▌       | 28/46 [08:09<01:09,  3.86s/itStatistiques pour le tick: 0.01                                             
Min price: 5.05
Max price: 16.595
Average spread: 17.715899546309238
Trades at bid: 56.061408152461624%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/GCMG_stats.json
Processing stocks:  63%|███████████▉       | 29/46 [08:15<01:18,  4.59s/itStatistiques pour le tick: 0.01                                             
Min price: 9.705
Max price: 24.990000000000002
Average spread: 6.977207166228927
Trades at bid: 55.25376718699791%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/CXW_stats.json
Processing stocks:  65%|████████████▍      | 30/46 [08:17<01:02,  3.88s/itStatistiques pour le tick: 0.01                                             
Min price: 8.955
Max price: 1076.885
Average spread: 140.7715647028694
Trades at bid: 69.10331384015595%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/LBTYB_stats.json
Processing stocks:  67%|████████████▊      | 31/46 [08:24<01:09,  4.65s/itStatistiques pour le tick: 0.01                                             
Min price: 5.835
Max price: 16.240000000000002
Average spread: 21.751961028392643
Trades at bid: 58.102497788062344%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/LION_stats.json
Processing stocks:  70%|█████████████▏     | 32/46 [08:27<00:57,  4.12s/itStatistiques pour le tick: 0.01                                             
Min price: 6.01
Max price: 35.34
Average spread: 34.20813039095344
Trades at bid: 62.06216977596151%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/RIOT_stats.json
Processing stocks:  72%|█████████████▋     | 33/46 [08:28<00:43,  3.32s/itStatistiques pour le tick: 0.01                                             
Min price: 4.185
Max price: 7.904999999999999
Average spread: 6.5101998960216365
Trades at bid: 60.95969193838767%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/HL_stats.json
Processing stocks:  74%|██████████████     | 34/46 [08:29<00:31,  2.66s/itStatistiques pour le tick: 0.01                                             
Min price: 5.885
Max price: 7.5
Average spread: 2.725320764598065
Trades at bid: 63.9707871246957%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/CGAU_stats.json
Processing stocks:  76%|██████████████▍    | 35/46 [08:30<00:23,  2.16s/itStatistiques pour le tick: 0.01                                             
Min price: 6.914999999999999
Max price: 10.745000000000001
Average spread: 5.3892050353509235
Trades at bid: 56.80535966149506%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/ASAI_stats.json
Processing stocks:  78%|██████████████▊    | 36/46 [08:31<00:18,  1.83s/itStatistiques pour le tick: 0.01                                             
Min price: 5.505
Max price: 7.625
Average spread: 10.953814167615464
Trades at bid: 67.59813870600465%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/CX_stats.json
Processing stocks:  80%|███████████████▎   | 37/46 [08:32<00:13,  1.54s/itStatistiques pour le tick: 0.01                                             
Min price: 6.515000000000001
Max price: 8.875
Average spread: 2.5609263288778386
Trades at bid: 70.50823748834317%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/YMM_stats.json
Processing stocks:  83%|███████████████▋   | 38/46 [08:33<00:10,  1.37s/itStatistiques pour le tick: 0.01                                             
Min price: 5.6
Max price: 7.86
Average spread: 4.910771050491613
Trades at bid: 70.58722062048743%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/ERIC_stats.json
Processing stocks:  85%|████████████████   | 39/46 [08:34<00:08,  1.25s/itStatistiques pour le tick: 0.01                                             
Min price: 5.3100000000000005
Max price: 7.145
Average spread: 2.4921725047984697
Trades at bid: 75.15595009596929%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/ITUB_stats.json
Processing stocks:  87%|████████████████▌  | 40/46 [08:35<00:07,  1.17s/itStatistiques pour le tick: 0.01                                             
Min price: 4.4399999999999995
Max price: 8.395
Average spread: 29.921738100876155
Trades at bid: 57.56571157944589%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/DBI_stats.json
Processing stocks:  89%|████████████████▉  | 41/46 [08:36<00:05,  1.13s/itStatistiques pour le tick: 0.01                                             
Min price: 5.995
Max price: 8.39
Average spread: 13.443382312776533
Trades at bid: 59.469055552520885%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/UA_stats.json
Processing stocks:  91%|█████████████████▎ | 42/46 [08:37<00:04,  1.11s/itStatistiques pour le tick: 0.01                                             
Min price: 4.220000000000001
Max price: 9.02
Average spread: 13.480982754547602
Trades at bid: 50.66146940703993%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/APLT_stats.json
Processing stocks:  93%|█████████████████▊ | 43/46 [08:38<00:03,  1.10s/itStatistiques pour le tick: 0.01                                             
Min price: 4.55
Max price: 8.54
Average spread: 37.41607737950093
Trades at bid: 60.9119726895879%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/PBI_stats.json
Processing stocks:  96%|██████████████████▏| 44/46 [08:39<00:02,  1.09s/itStatistiques pour le tick: 0.01                                             
Min price: 6.324999999999999
Max price: 12.625
Average spread: 17.67088259554578
Trades at bid: 50.50866098432775%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/IE_stats.json
Processing stocks:  98%|██████████████████▌| 45/46 [08:40<00:01,  1.07s/itStatistiques pour le tick: 0.01                                             
Min price: 5.5
Max price: 7.98
Average spread: 14.878496503496503
Trades at bid: 52.62237762237763%
Nombre de sauts: 0
Fichier JSON créé avec succès: results/summarystats/ANGO_stats.json
Processing stocks: 100%|███████████████████| 46/46 [08:41<00:00, 11.35s/it]

Traitement terminé. Consultez results/summarystats/processing_issues.txt pour les détails des problèmes rencontrés.
(HFT) janis@pc-samsung-janis:/c/Users/janis/AppData/Local/Programs/cursor$  
import numpy as np
# from taker import Model
# from taker.eval import evaluate_all;

# m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq4", limit=1000, device_map="cuda")
# results = evaluate_all(m, 1e5, ["pile"])

# print(results)
# loss = results["loss_data"]["pile"]["loss"]
# print(f"loss = {loss}, PPL = {np.exp(loss)}")

nan = None

full_results = {
    "fp16":   {'time':  53, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1137, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.50202964896908, 'topk': 84.85542104039628, 'skip': 55.173305999781064, 'topk_skip': 83.53916427000507}}},
    "bf16":   {'time':  55, 'loss_data': {'pile': {'perplexity': 289484.125, 'loss': 2.1142, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.519017900252145, 'topk': 84.87777400261082, 'skip': 55.15539323494581, 'topk_skip': 83.55608188123837}}},
    "bnb8":   {'time':  74, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1168, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.248994116700345, 'topk': 84.72756209652903, 'skip': 54.90063391284445, 'topk_skip': 83.41576522336223}}},
    "hqq8":   {'time': 122, 'loss_data': {'pile': {'perplexity': 314043.0, 'loss': 2.116, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.49040610861751, 'topk': 84.8491622109762, 'skip': 55.12255316608118, 'topk_skip': 83.53120304118941}}},
    "qint8":  {'time':  78, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1141, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.49755905652617, 'topk': 84.87777400261082, 'skip': 55.173305999781064, 'topk_skip': 83.56503826365599}}},
    "nf4" :   {'time':  95, 'loss_data': {'pile': {'perplexity': 1226090.875, 'loss': 2.1399, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.797464279966384, 'topk': 84.50492659287208, 'skip': 54.26473076119299, 'topk_skip': 83.13214644680406}}},
    "bnb4":   {'time': 277, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1556, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.19035782621913, 'topk': 84.04534968974089, 'skip': 53.918417307711444, 'topk_skip': 82.63954541383463}}},
    "hqq4":   {'time': 130, 'loss_data': {'pile': {'perplexity': 206663.5156, 'loss': 2.138, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.84842903381556, 'topk': 84.42535004738828, 'skip': 54.505557932866935, 'topk_skip': 83.07044692348263}}},
    "hqq4_1": {'time': 238, 'loss_data': {'pile': {'perplexity': 216561.6562, 'loss': 2.135, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.73487598576563, 'topk': 84.44680889111424, 'skip': 54.31050782688308, 'topk_skip': 83.06149054106501}}},
    # "gptq4":  {'time': 165, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1991, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 57.19156393417023, 'topk': 82.69409284530698, 'skip': 55.28491865894752, 'topk_skip': 81.03108658047614}}},
    # "awq4":   {'time': 177, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.183, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 57.51082269860411, 'topk': 83.08882854589778, 'skip': 55.634880163182046, 'topk_skip': 81.52303246642869}}},
    "gptq4":  {'time': 233, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1488, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.214191052688136, 'topk': 84.03148393567105, 'skip': 53.805902462847975, 'topk_skip': 82.64046007714465}}},
    "awq4":   {'time': 270, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1381, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 55.546402091765465, 'topk': 84.31265446470145, 'skip': 54.156741186672114, 'topk_skip': 82.9404670540511}}},
    "hqq3":   {'time': 201, 'loss_data': {'pile': {'perplexity': 152351.5156, 'loss': 2.1829, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 54.4920512866365, 'topk': 83.60365515638132, 'skip': 52.96306984983132, 'topk_skip': 82.1131091583986}}},
    "qfp8":   {'time': 350, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1113, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.466264909425796, 'topk': 84.86078575132777, 'skip': 55.15141262053798, 'topk_skip': 83.54712549882075}}},
    "qint8":  {'time':  79, 'loss_data': {'pile': {'perplexity': nan, 'loss': 2.1141, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 56.49755905652617, 'topk': 84.87777400261082, 'skip': 55.173305999781064, 'topk_skip': 83.56503826365599}}},

}

results_non_it = {
    "fp16": {'time':  54, 'loss_data': {'pile': {'perplexity': nan, 'loss': 1.8346, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.9545966631498, 'topk': 86.31730476922802, 'skip': 57.604466249365586, 'topk_skip': 85.07269597062306}}},
    "bf16": {'time':  54, 'loss_data': {'pile': {'perplexity': 63953.9492, 'loss': 1.8349, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.92151427907226, 'topk': 86.31998712469377, 'skip': 57.50495088916974, 'topk_skip': 85.06971050981718}}},
    "bnb8": {'time':  74, 'loss_data': {'pile': {'perplexity': nan, 'loss': 1.843, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.77309060996763, 'topk': 86.22521056490406, 'skip': 57.39150337854648, 'topk_skip': 84.96521938161155}}},
    "hqq8": {'time': 121, 'loss_data': {'pile': {'perplexity': 63880.6992, 'loss': 1.8352, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.88664365801756, 'topk': 86.30568122887645, 'skip': 57.4701205131012, 'topk_skip': 85.0527928985839}}},
    "hqq4": {'time': 131, 'loss_data': {'pile': {'perplexity': 91617.3906, 'loss': 1.8763, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.021136961070084, 'topk': 85.94892795193219, 'skip': 56.69489585717555, 'topk_skip': 84.6696587618299}}},
    "nf4" : {'time':  94, 'loss_data': {'pile': {'perplexity': 60465.3906, 'loss': 1.8797, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 58.04080756781889, 'topk': 85.87113964342555, 'skip': 56.51676336242499, 'topk_skip': 84.57213370883795}}},
    "int4": {'time': 279, 'loss_data': {'pile': {'perplexity': nan, 'loss': 1.9155, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 57.32104218451029, 'topk': 85.57697466068204, 'skip': 56.02913809746534, 'topk_skip': 84.27259247464846}}},
    "awq4": {'time': 180, 'loss_data': {'pile': {'perplexity': nan, 'loss': 1.9898, 'log_loss': nan}}, 'accuracy': {'pile': {'base': 59.416832224313985, 'topk': 84.38928748037166, 'skip': 57.55766865644779, 'topk_skip': 82.86088530261671}}},
    "hqq3": {'time': 201, 'loss_data': {'pile': {'perplexity': 180868.8281, 'loss': 1.9781, 'log_loss': -0.9231}}, 'accuracy': {'pile': {'base': 56.344664794978634, 'topk': 84.97344468088912, 'skip': 54.62497636510195, 'topk_skip': 83.55807218844228}}},
}

results_wmdp = {
    'fp16': {'time': 498.6996719837189, 'wmdp': {'acc,none': 0.5493456924754635, 'acc_stderr,none': 0.008006670553937022, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.7030636292223095, 'acc_stderr,none': 0.012811071595083552, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4877450980392157, 'acc_stderr,none': 0.024776634460297257, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4635128334172119, 'acc_stderr,none': 0.011189764031063174, 'alias': ' - wmdp_cyber'}},
    'bf16': {'time': 498.080846786499, 'wmdp': {'acc,none': 0.5498909487459106, 'acc_stderr,none': 0.007996955204187532, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.7069913589945012, 'acc_stderr,none': 0.012761558372411311, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.47549019607843135, 'acc_stderr,none': 0.024754284840506475, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.46451937594363363, 'acc_stderr,none': 0.011191393734217635, 'alias': ' - wmdp_cyber'}},
    "bnb8": {'time': 1954.130248785019, 'wmdp': {'acc,none': 0.549618320610687, 'acc_stderr,none': 0.007985430751989408, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.7109190887666929, 'acc_stderr,none': 0.012710898173814635, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4681372549019608, 'acc_stderr,none': 0.024733705353784115, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.463009562154001, 'acc_stderr,none': 0.011188931993180416, 'alias': ' - wmdp_cyber'}},
    "hqq8": {'time': 1675.30868601799, 'wmdp': {'acc,none': 0.5466194111232279, 'acc_stderr,none': 0.007994795752669672, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.7062058130400628, 'acc_stderr,none': 0.012771552343692495, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.47794117647058826, 'acc_stderr,none': 0.024759948652192453, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4584801207851032, 'acc_stderr,none': 0.011180927591557187, 'alias': ' - wmdp_cyber'}},
    "hqq4": {'time': 1854.859544992447, 'wmdp': {'acc,none': 0.542257360959651, 'acc_stderr,none': 0.008029350490447059, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6920659858601729, 'acc_stderr,none': 0.012943717578228237, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.46568627450980393, 'acc_stderr,none': 0.024725647848553352, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4620030196275793, 'acc_stderr,none': 0.01118723353202166, 'alias': ' - wmdp_cyber'}},
    "nf4":  {'time': 1232.5785446166992, 'wmdp': {'acc,none': 0.5441657579062159, 'acc_stderr,none': 0.008005450487303053, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.7014925373134329, 'acc_stderr,none': 0.01283055871562815, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.47058823529411764, 'acc_stderr,none': 0.02474116366703947, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4584801207851032, 'acc_stderr,none': 0.011180927591557182, 'alias': ' - wmdp_cyber'}},
    "bnb4": {'time': 1625.913871049881, 'wmdp': {'acc,none': 0.5272628135223555, 'acc_stderr,none': 0.008023942507716299, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6849960722702279, 'acc_stderr,none': 0.01302442230773214, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.46568627450980393, 'acc_stderr,none': 0.024725647848553352, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.43885254151987924, 'acc_stderr,none': 0.011135460634635956, 'alias': ' - wmdp_cyber'}},
    "gptq4":{'time': 2610.4708154201508, 'wmdp': {'acc,none': 0.5329880043620502, 'acc_stderr,none': 0.008028267842384633, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6865671641791045, 'acc_stderr,none': 0.013006792288528305, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4852941176470588, 'acc_stderr,none': 0.024773357777817893, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4443885254151988, 'acc_stderr,none': 0.011150065004044201, 'alias': ' - wmdp_cyber'}},
    "awq4": {'time': 3161.272638320923, 'wmdp': {'acc,none': 0.5455288985823337, 'acc_stderr,none': 0.008022694629545654, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6959937156323645, 'acc_stderr,none': 0.012897346986972721, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4681372549019608, 'acc_stderr,none': 0.024733705353784115, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4650226472068445, 'acc_stderr,none': 0.011192191404494253, 'alias': ' - wmdp_cyber'}},
    # {'time': 3569.5381786823273, 'wmdp': {'acc,none': 0.5455288985823337, 'acc_stderr,none': 0.008022694629545654, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6959937156323645, 'acc_stderr,none': 0.012897346986972721, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4681372549019608, 'acc_stderr,none': 0.024733705353784115, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4650226472068445, 'acc_stderr,none': 0.011192191404494253, 'alias': ' - wmdp_cyber'}}
    "hqq3": {'time': 2919.9338159561157, 'wmdp': {'acc,none': 0.5122682660850599, 'acc_stderr,none': 0.008091095281557784, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6488609583660644, 'acc_stderr,none': 0.01338356541328383, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4411764705882353, 'acc_stderr,none': 0.024611966106856852, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4393558127830901, 'acc_stderr,none': 0.011136846353673803, 'alias': ' - wmdp_cyber'}},
    #"hqq3": {'time': 3073.138346195221, 'wmdp': {'acc,none': 0.5122682660850599, 'acc_stderr,none': 0.008091095281557784, 'alias': 'wmdp'}, 'wmdp_bio': {'acc,none': 0.6488609583660644, 'acc_stderr,none': 0.01338356541328383, 'alias': ' - wmdp_bio'}, 'wmdp_chem': {'acc,none': 0.4411764705882353, 'acc_stderr,none': 0.024611966106856852, 'alias': ' - wmdp_chem'}, 'wmdp_cyber': {'acc,none': 0.4393558127830901, 'acc_stderr,none': 0.011136846353673803, 'alias': ' - wmdp_cyber'}}
}

results_minerva_100 = {
    "fp16":  {'time': 674.9262909889221, 'minerva_math_algebra': {'exact_match,none': 0.39, 'exact_match_stderr,none': 0.04902071300001975, 'alias': 'minerva_math_algebra'}},
    "bfp16": {'time': 686.0503370761871, 'minerva_math_algebra': {'exact_match,none': 0.38, 'exact_match_stderr,none': 0.04878317312145632, 'alias': 'minerva_math_algebra'}},
    "hqq4_1": {'time': 574.5966403484344, 'minerva_math_algebra': {'exact_match,none': 0.36, 'exact_match_stderr,none': 0.04824181513244218, 'alias': 'minerva_math_algebra'}},
    "nf4"    :{'time': 703.0646848678589, 'minerva_math_algebra': {'exact_match,none': 0.34, 'exact_match_stderr,none': 0.04760952285695235, 'alias': 'minerva_math_algebra'}},
    "bnb4": {'time': 712.2663633823395, 'minerva_math_algebra': {'exact_match,none': 0.34, 'exact_match_stderr,none': 0.04760952285695235, 'alias': 'minerva_math_algebra'}},
}

results_minerva_algebra = {
    "fp16": {'time': 8116.727879047394, 'minerva_math_algebra': {'exact_match,none': 0.37489469250210616, 'exact_match_stderr,none': 0.014056878617075412, 'alias': 'minerva_math_algebra'}},
    "bfp16": {'time': 8099.244785070419, 'minerva_math_algebra': {'exact_match,none': 0.37152485256950296, 'exact_match_stderr,none': 0.014031226813047564, 'alias': 'minerva_math_algebra'}},
    "hqq8": {'time': 54041.58850693703, 'minerva_math_algebra': {'exact_match,none': 0.37910699241786017, 'exact_match_stderr,none': 0.014087921965045581, 'alias': 'minerva_math_algebra'}},
    # "hqq8": {'time': 53559.62073945999, 'minerva_math_algebra': {'exact_match,none': 0.37910699241786017, 'exact_match_stderr,none': 0.014087921965045581, 'alias': 'minerva_math_algebra'}},
    "bnb8": {'time': 19918.836363554, 'minerva_math_algebra': {'exact_match,none': 0.36310025273799496, 'exact_match_stderr,none': 0.013963891618729737, 'alias': 'minerva_math_algebra'}},
    "hqq4_1": {'time': 6578.001960992813, 'minerva_math_algebra': {'exact_match,none': 0.33698399326032014, 'exact_match_stderr,none': 0.013725377510777646, 'alias': 'minerva_math_algebra'}},
    # "hqq4_1": {'time': 6529.291163444519, 'minerva_math_algebra': {'exact_match,none': 0.33698399326032014, 'exact_match_stderr,none': 0.013725377510777646, 'alias': 'minerva_math_algebra'}},
    "nf4": {'time': 8403.131981372833, 'minerva_math_algebra': {'exact_match,none': 0.3125526537489469, 'exact_match_stderr,none': 0.013459811280854548, 'alias': 'minerva_math_algebra'}},
    # "nf4": {'time': 8646.41929268837, 'minerva_math_algebra': {'exact_match,none': 0.3125526537489469, 'exact_match_stderr,none': 0.013459811280854548, 'alias': 'minerva_math_algebra'}},
    "int4": {'time': 8514.2936835289, 'minerva_math_algebra': {'exact_match,none': 0.2931760741364785, 'exact_match_stderr,none': 0.013218358882377854, 'alias': 'minerva_math_algebra'}},
    # "int4": {'time': 8000.429564237595, 'minerva_math_algebra': {'exact_match,none': 0.2931760741364785, 'exact_match_stderr,none': 0.013218358882377854, 'alias': 'minerva_math_algebra'}},
}

print("Llama 3 8B Instruct")
for k,v in full_results.items():
    print(f'{k} - Acc={v["accuracy"]["pile"]["base"]:.2f}% PPL={np.exp(v["loss_data"]["pile"]["loss"]):.3f} T={v["time"]}s')

print("\nLlama 3 8B")
for k,v in results_non_it.items():
    print(f'{k} - Acc={v["accuracy"]["pile"]["base"]:.2f}% PPL={np.exp(v["loss_data"]["pile"]["loss"]):.3f} T={v["time"]}s')

print("\nLlama 3 8B Instruct WMDP")
for k,v in results_wmdp.items():
    print(f'{k} - Acc={v["wmdp"]["acc,none"]*100:.2f}% T={v["time"]:.2f}s')

print("\nLlama 3 8B Instruct Math Algebra 100")
for k,v in results_minerva_100.items():
    print(f'{k} - Acc={v["minerva_math_algebra"]["exact_match,none"]*100:.2f}% T={v["time"]:.2f}s')

print("\nLlama 3 8B Instruct Math Algebra")
for k,v in results_minerva_algebra.items():
    print(f'{k} - Acc={v["minerva_math_algebra"]["exact_match,none"]*100:.2f}% T={v["time"]/3600:.2f}h')

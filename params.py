BATCH_SIZE = 30
LEARN_RATE = 5e-5
EPOCH = 10
NUM_LABELS = 31
model_name = 'xlnet-base-cased'
hidden_layer_size = 768
save_path = './results/trained_model_xlnet.bin'

LabeltoTeamsDict = {
    0: 'Irrelavant',
    1: 'Milwaukee Bucks',
    2: 'Boston Celtics',
    3: 'Philadelphia 76ers',
    4: 'Cleveland Cavaliers',
    5: 'New York Knicks',
    6: 'Brooklyn Nets',
    7: 'Atlanta Hawks',
    8: 'Miami Heat',
    9: 'Toronto Raptors',
    10: 'Chicago Bulls',
    11: 'Indiana Pacers',
    12: 'Washigton Wizards', 
    13: 'Orlando Magic',
    14: 'Charlotte Hornets',
    15: 'Detroit Pistons',
    16: 'Denver Nuggets',
    17: 'Memphis Grizzlies',
    18: 'Sacramento Kings',
    19: 'Phoenix Suns',
    20: 'LA Clippers',
    21: 'Golden State Warriors',
    22: 'Los Angeles Lakers',
    23: 'Minnesota Timberwolves',
    24: 'New Orleans Pelicans',
    25: 'Oklahoma City Thunder',
    26: 'Dallas Mavericks',
    27: 'Utah Jazz',
    28: 'Portland Trail Blazers',
    29: 'Houston Rockets',
    30: 'San Antonio Spurs'
    # 31: 'Negative Milwaukee Bucks',
    # 32: 'Negative Boston Celtics',
    # 33: 'Negative Philadelphia 76ers',
    # 34: 'Negative Cleveland Cavaliers',
    # 35: 'Negative New York Knicks',
    # 36: 'Negative Brooklyn Nets',
    # 37: 'Negative Atlanta Hawks',
    # 38: 'Negative Miami Heat',
    # 39: 'Negative Toronto Raptors',
    # 40: 'Negative Chicago Bulls',
    # 41: 'Negative Indiana Pacers',
    # 42: 'Negative Washigton Wizards', 
    # 43: 'Negative Orlando Magic',
    # 44: 'Negative Charlotte Hornets',
    # 45: 'Negative Detroit Pistons',
    # 46: 'Negative Denver Nuggets',
    # 47: 'Negative Memphis Grizzlies',
    # 48: 'Negative Sacramento Kings',
    # 49: 'Negative Phoenix Suns',
    # 50: 'Negative LA Clippers',
    # 51: 'Negative Golden State Warriors',
    # 52: 'Negative Los Angeles Lakers',
    # 53: 'Negative Minnesota Timberwolves',
    # 54: 'Negative New Orleans Pelicans',
    # 55: 'Negative Oklahoma City Thunder',
    # 56: 'Negative Dallas Mavericks',
    # 57: 'Negative Utah Jazz',
    # 58: 'Negative Portland Trail Blazers',
    # 59: 'Negative Houston Rockets',
    # 60: 'Negative San Antonio Spurs'
}

synonyms = [
    ['嘴綠', 'Green', 'Draymond Green', '格林', '綠師傅', '綠seafood', '母獅', '姆斯', '姆姆', '姆', '喇叭'],
    ['Lebron', 'Lebron James', '18詹', 'LBJ', 'lbj', '喇叭詹', '老詹', '詹皇', '老漢'],
    ['勇', '勇士'],
    ['湖人', '湖', 'LA'],
    ['G', 'Game'],
    ['迷', '球迷'],
    ['AD', 'Antony Davis', '安東尼', '戴維斯', '玻璃人', '一眉哥'], 
    ['哭狸', '庫裡', '庫里', 'Curry', '咖哩', 'SC', '柯瑞', '咖哩小子'], 
    ['魯尼', 'Looney', 'Kevon Looney', 'KL'], 
    ['K湯', 'k湯', 'Klay', 'Klay Tompson', '球佛'], 
    ['MB', 'Embiid', 'Joel Embiid', '恩彼得'], 
    ['dlo', 'DLO', 'D Angelo Russell', 'Russell', 'DLo', 'Russ', 'DR'],
    ['普爾', 'Poole', 'poole'], 
    ['艾頓', '艾盾', 'Ayton', '愛頓', 'DA', '頓寶', '軟頓', '破頓', '欸疼'], 
    ['AE', 'Anthony Edwards'], 
    ['書人', 'Booker', 'DB'],
    ['阿肥', 'Jokic', '小丑', '肥'],
    ['KD', 'kd', '杜蘭特', '死神KD', 'Kevin Durant'],
    ['內線', '禁區'],
    ['外線', '外圍'],
    ['Murray', '穆瑞', 'JM'], 
    ['阿銀', 'Adam Silver'], 
    ['雞塊', '金塊', '金'],
    ['總冠', '總冠軍', '冠軍'],
    ['吉八', '吉巴', 'Jimmy Butler', '老巴', '黑巴', '黑八', '黑8'],
    ['青賽', '賽爾提克', '綠軍'],
    ['Brown', '布朗', '杰倫']
]
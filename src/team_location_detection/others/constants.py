from os.path import join
from team_location_detection.constants import configs_dir, data_dir, soccernet_dir

ball_action_dir = join(data_dir, "team_location_detection")
configs_dir = join(configs_dir, "team_location_detection")
experiments_dir = join(ball_action_dir, "experiments")
predictions_dir = join(ball_action_dir, "predictions")
soccernet_dir = soccernet_dir

fold_games = [
    "england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
    "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
    "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
    "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest",
    "england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End",
    "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
    "england_efl/2019-2020/2019-10-01 - Reading - Fulham",
]
fold2games = {fold: [game] for fold, game in enumerate(fold_games)}
game2fold = {game: fold for fold, games in fold2games.items() for game in games}
folds = sorted(fold2games.keys())
challenge_games = [
    "england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
    "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City",
]

# classes = [
#     "PASS",
#     "DRIVE",
# ]

labels_filename = "Labels-ball-location-team.json"
videos_extension = "mp4"

classes = [
    "PASS",
    "DRIVE",
    "HEADER",
    "HIGH PASS",
    "OUT",
    "CROSS",
    "THROW IN",
    "SHOT",
    "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE",
    "FREE KICK",
    "GOAL",
]


num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

location_classes = [
    "Right center midfield",
    "Right bottom midfield",
    "Right top midfield",
    "Right bottom corner",
    "Right top corner",
    "Right bottom box",
    "Right edge of the box",
    "Right top box",
    "0",
    "Left center midfield",
    "Left bottom midfield",
    "Left top midfield",
    "Left bottom corner",
    "Left top corner",
    "Left bottom box",
    "Left edge of the box",
    "Left top box",
]


num_location_classes = len(location_classes)
target2location: dict[int, str] = {trg: cls for trg, cls in enumerate(location_classes)}
location_class2target: dict[str, int] = {
    cls: trg for trg, cls in enumerate(location_classes)
}

location_easy_classes = [
    "Right",
    "Left",
    "0",
]

num_location_easy_classes = len(location_easy_classes)
target2location_easy: dict[int, str] = {
    trg: cls for trg, cls in enumerate(location_easy_classes)
}
location_easy_class2target: dict[str, int] = {
    cls: trg for trg, cls in enumerate(location_easy_classes)
}

location_hard_classes = [
    "center midfield",
    "bottom midfield",
    "top midfield",
    "bottom corner",
    "top corner",
    "bottom box",
    "edge of the box",
    "top box",
]

num_location_hard_classes = len(location_hard_classes)
target2location_hard: dict[int, str] = {
    trg: cls for trg, cls in enumerate(location_hard_classes)
}
location_hard_class2target: dict[str, int] = {
    cls: trg for trg, cls in enumerate(location_hard_classes)
}

team_classes = [
    "right",
    "left",
]

num_team_classes = len(team_classes)
target2team: dict[int, str] = {trg: cls for trg, cls in enumerate(team_classes)}
team_class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(team_classes)}

num_halves = 1
halves = list(range(1, num_halves + 1))
postprocess_params = {
    "gauss_sigma": 3.0,
    "height": 0.5,
    "distance": 15,
}

video_fps = 25.0

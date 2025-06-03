import json
import os
import subprocess
import sys
import argparse
import json
from tqdm import tqdm

RESLUT_DIR = "../../../data/sample/results"
VIDEO_DIR = "../../../data/videos"
MATCH_INFO_DIR = os.path.join(RESLUT_DIR, "video_info", "sample_game_info.json")
key_path = "../../../OpenAI_API_key.txt"
with open(key_path) as f:
    key = f.read().strip()
os.environ["OPENAI_API_KEY"] = key
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        help="result path",
        required=True,
        choices=["action", "team", "location", "all"],
    )
    parser.add_argument(
        "--name",
        action="store_true",
        help="use name",
    )
    return parser.parse_args()


def location_team(location, team):
    if team == "right":  # 右が自陣
        if location == "Left top midfield":
            return "right side opponent midfield"
        elif location == "Left bottom midfield":
            return "left side opponent midfield"
        elif location == "Left center midfield":
            return "opponent center midfield"
        elif location == "Left top corner":
            return "right side corner"
        elif location == "Left bottom corner":
            return "left side corner"
        elif location == "Left edge of the box":
            return "edge of the box"
        elif location == "Left top box":
            return "right side opponent box"
        elif location == "Left bottom box":
            return "left side opponent box"
        elif location == "Right top midfield":
            return "right side own midfield"
        elif location == "Right bottom midfield":
            return "left side own midfield"
        elif location == "Right center midfield":
            return "own center midfield"
        elif location == "Right top corner":
            return "right side own corner"
        elif location == "Right bottom corner":
            return "left side own corner"
        elif location == "Right edge of the box":
            return "own edge of the box"
        elif location == "Right top box":
            return "right side own box"
        elif location == "Right bottom box":
            return "left side own box"
        elif location == "OUT":
            return "OUT"

    else:
        if location == "Left top midfield":
            return "left side own midfield"
        elif location == "Left bottom midfield":
            return "right side own midfield"
        elif location == "Left center midfield":
            return "own center midfield"
        elif location == "Left top corner":
            return "left side corner"
        elif location == "Left bottom corner":
            return "right side corner"
        elif location == "Left edge of the box":
            return "own edge of the box"
        elif location == "Left top box":
            return "left side own box"
        elif location == "Left bottom box":
            return "right side own box"
        elif location == "Right top midfield":
            return "left side opponent midfield"
        elif location == "Right bottom midfield":
            return "right side opponent midfield"
        elif location == "Right center midfield":
            return "opponent center midfield"
        elif location == "Right top corner":
            return "left side opponent corner"
        elif location == "Right bottom corner":
            return "right side opponent corner"
        elif location == "Right edge of the box":
            return "opponent edge of the box"
        elif location == "Right top box":
            return "left side opponent box"
        elif location == "Right bottom box":
            return "right side opponent box"
        elif location == "OUT":
            return "OUT"


def making_event_text(
    action,
    previous_action,
    team,
    team_name,
    location,
    previous_team,
    previous_team_name,
    next_team=None,
    next_team_name=None,
    player_name=None,
):
    if player_name is not None:
        player = player_name
    else:
        player = "Player"

    if action == "GOAL":
        return "GOAL!!"
    if location == "OUT":
        return "OUT"

    if action == "PLAYER SUCCESSFUL TACKLE":
        if next_team == team:
            return f"{player}({team_name}) tackles {next_team_name} at {location} and gets the ball"
        else:
            return f"{player}({team_name}) tackles {previous_team_name} at {location}, but can't get the ball"

    elif action == "BALL PLAYER BLOCK":
        if next_team == team:
            return (
                f"{player}({team_name}) blocks the ball at {location} and gets the ball"
            )
        else:
            return f"{player}({team_name}) blocks the ball at {location}, but can't get the ball"

    if team != previous_team:
        if action == "DRIVE":
            return f"{player}({team_name}) gets the opponent ball at {location}"
        elif action == "PASS":
            return f"{player}({team_name}) gets the ball and passes from {location}"
        elif action == "CROSS":
            return f"{player}({team_name}) gets the ball and crosses from {location}"
        elif action == "SHOT":
            return f"{player}({team_name}) gets the ball and shots from {location}"
        elif action == "HIGH PASS":
            return f"{player}({team_name}) gets the ball and tries high passes from {location}"
        elif action == "FREE KICK":
            return f"{player}({team_name}) gets the ball and takes a free kick from {location}"
        elif action == "THROW IN":
            return f"{player}({team_name}) takes a throw in from {location}"
        elif action == "HEADER":
            return f"{player}({team_name}) heads the ball at {location}"
        else:
            return str(action)
    else:
        if action == "DRIVE":
            return f"{player}({team_name}) receives the ball at {location}"
        elif action == "PASS":
            if previous_team == team and previous_action == "PASS":
                return f"{player}({team_name}) directly passes from {location}"
            else:
                return f"{player}({team_name}) passes from {location}"
        elif action == "CROSS":
            return f"{player}({team_name}) crosses from {location}"
        elif action == "SHOT":
            return f"{player}({team_name}) shots from {location}"
        elif action == "HIGH PASS":
            return f"{player}({team_name}) tries high passes from {location}"
        elif action == "FREE KICK":
            return f"{player}({team_name}) takes a free kick from {location}"
        elif action == "THROW IN":
            return f"{player}({team_name}) takes a throw in from {location}"
        elif action == "HEADER":
            return f"{player}({team_name}) heads the ball at {location}"
        else:
            return str(action)


def making_srt(task, name: bool):
    with open(MATCH_INFO_DIR, "r") as f:
        video_dicts = json.load(f)

    for video_dict in tqdm(video_dicts):
        t = 1
        video_name = video_dict["video"]
        result_path = os.path.join(
            RESLUT_DIR, video_name, "results_spotting_my_filtered_integrated_name.json"
        )
        if name:
            result_path = os.path.join(
                RESLUT_DIR,
                video_name,
                "results_spotting_my_filtered_integrated_name.json",
            )
        with open(result_path, "r") as f:
            results = json.load(f)
            results = results["predictions"]

        srt_dir = os.path.join(RESLUT_DIR, video_name, "srt")
        if os.path.exists(srt_dir):
            # すでにsrt_dirが存在する場合は、空にする
            for filename in os.listdir(srt_dir):
                file_path = os.path.join(srt_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        os.makedirs(srt_dir, exist_ok=True)
        output_path = os.path.join(srt_dir, "event_text.srt")
        if task in ["action", "team", "location"]:
            output_path = os.path.join(srt_dir, f"event_text_{task}.srt")
        if name:
            output_path = os.path.join(srt_dir, f"event_text_{task}_name.srt")
        # if os.path.exists(output_path):
        #     print(f"{output_path} already exists.")
        #     continue

        with open(output_path, "w") as f:
            for i, result in enumerate(results):
                action = result["action"]
                team_side = result["team_own_side"]
                location = location_team(result["location"], team_side)

                team_name = result["team"]
                # もしvideo_dictが"second_start"を持っていたら、それを使う
                if "second_start" in video_dict.keys():
                    if result["frame"] <= video_dict["second_start"]:
                        match_dict = video_dict["halves"]["1st"]
                    else:
                        match_dict = video_dict["halves"]["2nd"]
                    team_name = match_dict[team_side]

                previous_action = None
                previous_team_side = None
                previous_team_name = None

                if not i == 0:
                    previous_action = results[i - 1]["action"]
                    previous_team_side = results[i - 1]["team_own_side"]
                    previous_team_name = results[i - 1]["team"]

                next_team = None
                next_team_name = None
                if i + 1 < len(results):
                    next_team = results[i + 1]["team_own_side"]
                    next_team_name = results[i + 1]["team"]

                player_name = None
                if name:
                    player_name = result["name"]

                event_text = making_event_text(
                    action,
                    previous_action,
                    team_side,
                    team_name,
                    location,
                    previous_team_side,
                    previous_team_name,
                    next_team,
                    next_team_name,
                    player_name,
                )

                f.write(f"{t}\n")
                start = result["position"]  # ms
                ## 00:00:00.000 (時:分:秒.ミリ秒)
                if len(str(start)) > 3:
                    start_second = str(start)[:-3]
                    start_millisecond = str(start)[-3:]
                else:
                    start_second = "0"
                    if len(str(start)) == 3:
                        start_millisecond = str(start)
                    elif len(str(start)) == 2:
                        start_millisecond = "0" + str(start)

                ##start_secondが2桁のとき
                if int(start_second) >= 10:
                    if int(start_second) >= 3600:
                        start_hour = "01:"
                        start_second = int(start_second) - 3600
                        start_minutes = str(int(start_second) // 60)
                        if len(start_minutes) == 1:
                            start_minutes = "0" + start_minutes
                        start_second = str(int(start_second) - int(start_minutes) * 60)
                        if len(str(start_second)) == 1:
                            start_second = "0" + start_second

                    else:
                        start_hour = "00:"
                        start_minutes = str(int(start_second) // 60)
                        if len(start_minutes) == 1:
                            start_minutes = "0" + start_minutes
                        start_second = str(int(start_second) - int(start_minutes) * 60)
                        if len(str(start_second)) == 1:
                            start_second = "0" + start_second

                    start2 = (
                        start_hour
                        + start_minutes
                        + ":"
                        + start_second
                        + "."
                        + start_millisecond
                    )

                else:
                    start2 = "00:00:0" + start_second + "." + start_millisecond
                # 次のフレームの開始時間まで1.5秒以上なら、startから1.5秒後までを1つの字幕とする
                # そうでない場合は、次のフレームの開始時間までの0.2秒前までを1つの字幕とする
                if i + 1 < len(results):
                    end = int(results[i + 1]["position"])
                    start = int(result["position"])
                    if end - start > 1500:
                        end = start + 1500

                        if len(str(end)) > 3:
                            end_second = str(end)[:-3]
                            end_millisecond = str(end)[-3:]
                        else:
                            end_second = "0"
                            if len(str(end)) == 3:
                                end_millisecond = str(end)
                            elif len(str(end)) == 2:
                                end_millisecond = "0" + str(end)

                        if int(end_second) >= 10:
                            if int(end_second) >= 3600:
                                end_hour = "01:"
                                end_second = int(end_second) - 3600
                                end_minutes = str(int(end_second) // 60)
                                if len(end_minutes) == 1:
                                    end_minutes = "0" + end_minutes
                                end_second = str(
                                    int(end_second) - int(end_minutes) * 60
                                )
                                if len(str(end_second)) == 1:
                                    end_second = "0" + end_second
                            else:
                                end_hour = "00:"
                                end_minutes = str(int(end_second) // 60)
                                if len(end_minutes) == 1:
                                    end_minutes = "0" + end_minutes
                                end_second = str(
                                    int(end_second) - int(end_minutes) * 60
                                )
                                if len(str(end_second)) == 1:
                                    end_second = "0" + end_second
                            end2 = (
                                end_hour
                                + end_minutes
                                + ":"
                                + end_second
                                + "."
                                + end_millisecond
                            )
                        else:
                            end2 = "00:00:0" + end_second + "." + end_millisecond
                    else:
                        end = results[i + 1]["position"]
                        end = end - 200
                        if len(str(end)) > 3:
                            end_second = str(end)[:-3]
                            end_millisecond = str(end)[-3:]
                        else:
                            end_second = "0"
                            if len(str(end)) == 3:
                                end_millisecond = str(end)
                            elif len(str(end)) == 2:
                                end_millisecond = "0" + str(end)

                        if int(end_second) >= 10:
                            if int(end_second) >= 3600:
                                end_hour = "01:"
                                end_second = int(end_second) - 3600
                                end_minutes = str(int(end_second) // 60)
                                if len(end_minutes) == 1:
                                    end_minutes = "0" + end_minutes
                                end_second = str(
                                    int(end_second) - int(end_minutes) * 60
                                )
                                if len(str(end_second)) == 1:
                                    end_second = "0" + end_second
                            else:
                                end_hour = "00:"
                                end_minutes = str(int(end_second) // 60)
                                if len(end_minutes) == 1:
                                    end_minutes = "0" + end_minutes
                                end_second = str(
                                    int(end_second) - int(end_minutes) * 60
                                )
                                if len(str(end_second)) == 1:
                                    end_second = "0" + end_second
                            end2 = (
                                end_hour
                                + end_minutes
                                + ":"
                                + end_second
                                + "."
                                + end_millisecond
                            )
                        else:
                            end2 = "00:00:0" + end_second + "." + end_millisecond
                else:
                    end = results[i]["position"] + 1500
                    if len(str(end)) > 3:
                        end_second = str(end)[:-3]
                        end_millisecond = str(end)[-3:]
                    else:
                        end_second = "0"
                        if len(str(end)) == 3:
                            end_millisecond = str(end)
                        elif len(str(end)) == 2:
                            end_millisecond = "0" + str(end)

                    if int(end_second) >= 10:
                        if int(end_second) >= 3600:
                            end_hour = "01:"
                            end_second = int(end_second) - 3600
                            end_minutes = str(int(end_second) // 60)
                            if len(end_minutes) == 1:
                                end_minutes = "0" + end_minutes
                            end_second = str(int(end_second) - int(end_minutes) * 60)
                            if len(str(end_second)) == 1:
                                end_second = "0" + end_second
                        else:
                            end_hour = "00:"
                            end_minutes = str(int(end_second) // 60)
                            if len(end_minutes) == 1:
                                end_minutes = "0" + end_minutes
                            end_second = str(int(end_second) - int(end_minutes) * 60)
                            if len(str(end_second)) == 1:
                                end_second = "0" + end_second
                        end2 = (
                            end_hour
                            + end_minutes
                            + ":"
                            + end_second
                            + "."
                            + end_millisecond
                        )
                    else:
                        end2 = "00:00:0" + end_second + "." + end_millisecond
                f.write(f"{start2} --> {end2}\n")
                if task == "all":
                    f.write(f"{event_text}\n\n")
                elif task == "action":
                    f.write(f"{action}\n\n")
                elif task == "team":
                    f.write(f"{team_name}\n\n")
                elif task == "location":
                    f.write(f"{location}\n\n")
                t += 1

        if task == "all":
            client = OpenAI()
            en_inst = "You are a passionate soccer commentator. Input is str file and you change only text part. Change the text more contextually connected passionate but very very short sentences, using many conjunctions. It's Ok to omit location information like right bottom midfield to make text shorter in some text. For the first text, put team name but it's OK not to include this team name if it is used many times in a row. But don't forget to include the word 'player'. Of course, use conjunctions as much as possible for each passionate text and don't forget to include time part."
            if name:
                en_inst = "You are a passionate soccer commentator. Input is str file and you change only text part. Change the text more contextually connected passionate but very very short sentences, using many conjunctions. It's Ok to often omit location or team information like right bottom midfield to make text shorter. For the first text, put player name but it's OK not to include this team name if it is used many times in a row. Of course, use conjunctions as much as possible for each passionate text and don't forget to include time part."
            save_path = os.path.join(srt_dir, "commentary.srt")
            jsonl_save_path = os.path.join(srt_dir, "commentary_en.jsonl")
            if name:
                save_path = os.path.join(srt_dir, "commentary_name.srt")
                jsonl_save_path = os.path.join(srt_dir, "commentary_name_en.jsonl")

            with open(output_path, "r") as f:
                srt_file = f.read()

            messages = [
                {"role": "system", "content": en_inst},
                {"role": "user", "content": srt_file},
            ]
            completion = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=1.0, seed=42
            )
            output = completion.choices[0].message.content

            with open(save_path, "w") as f:
                f.write(output)

            with open(save_path, "r") as f:
                srt = f.read()

            k = 0
            for i, line in enumerate(srt.split("\n")):
                if i % 4 == 0:  # 数字
                    json_dict = {}

                elif i % 4 == 1:  # 時間
                    json_dict["start_time"] = line[6:12]
                    json_dict["end_time"] = line[23:29]

                elif i % 4 == 2:  # テキスト
                    json_dict["text"] = line
                    json_dict["action"] = results[k]["action"]
                    json_dict["location"] = results[k]["location"]
                    json_dict["team"] = results[k]["team"]
                    if name:
                        json_dict["name"] = results[k]["name"]

                    with open(jsonl_save_path, "a") as f:
                        json.dump(json_dict, f)
                    k += 1
                    if k >= len(results):
                        break
                    else:
                        with open(jsonl_save_path, "a") as f:
                            f.write("\n")
    print("Finished")


if __name__ == "__main__":
    args = parse_args()
    making_srt(args.task, args.name)

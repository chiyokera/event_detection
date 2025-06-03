instruction = """You create a web text commentary for explaining one Soccer event in the match.
Event texts which represent the events of a soccer match with a timestamp and Reference text which will be base of the final web text comment are provided.
For making the web text commentary, please follow the rules below.

#Rules
1. Based on the timestamp for each event text and the content of the event texts and Reference text, please create one web text commentary for the soccer match, taking into account the context.
2. The Reference text is the base of the web text commentary for the soccer match. Based on the information in the event texts, please create a more detailed commentary based on the Reference Text.
3. The player names in the event texts were replaced with [Player], but please use them as they are for final web text commentary. For the team name, the format should be like [Player](Team Name), so the actual team name will be inserted after [Player].
4. Some event texts may omit team names, but you should infer those team names before or after the event text.
5. Some event texts include the location of the event on the field, and you can use that information. In general, use the location words like right side, but the location information is related to the box, please use the word "box".
6. After the Shot event, if there is no Goal and it goes OUT, and a relatively long time passes until the next event, consider that the shot missed and add a comment about it. However, if the Reference Text says "Goal" or something like that, please create a commentary about Goal based on the Reference text while considering the event texts context.
7. If the Reference text is about out-of-play event such as foul scene, offside, substitution, injury, VAR, or other scenes where the play is interrupted, the event text is not very relevant, so if there are no particularly important event texts, please output the Reference text almost as it is.
8. You mustn't need to include the timestamp in the web text commentary.
9. You don't need to make so many sentences.

Additionally, please add a version that behaves like a very enthusiastic commentator.
Also, please add a text from the opposite perspective of the team mentioned this time.
"""

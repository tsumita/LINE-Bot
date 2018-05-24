# PythonでLINEBot

- FLASK使ってサーバ立てる
- ローカルPCで動かすかHerokuでデプロイする

## PythonでのLINEBotAPI
### テキストメッセージに対する処理（おうむ返し）
``` python
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=event.message.text))
```
### テキストメッセージに対する処理（特定キーワードへの応答）
```python
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text == "おはよう":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="おはようございます"))
```
### テキストメッセージに対する処理（カルーセルで選択肢つけた応答）
```python
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    elif event.message.text == "こんにちは":
        date_picker = TemplateSendMessage(
            alt_text='Selection Menu',
            template=ButtonsTemplate(
                text='次から選んでください',
                title='Option',
                actions=[
                    PostbackTemplateAction(
                        label='日本語',
                        data='action=click&itemid=1',
                    ),
                    PostbackTemplateAction(
                        label='英語',
                        data='action=click&itemid=2',
                    )
                ]
            )
        )

        line_bot_api.reply_message(event.reply_token, date_picker)
```
### カルーセルレスポンスに相手が何かしらアクションした時の処理
```python
@handler.add(PostbackEvent)
def handle_message(event):
    event_data = event.postback.data
    if event_data == 'action=click&itemid=1':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="こんにちは"))

    elif event_data == 'action=click&itemid=2':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Hello!!"))
```
### スタンプに対しておうむ返し（デフォルトであるスタンプのみ返す）
```python
@handler.add(MessageEvent, message=StickerMessage)
def handle_message(event):
    line_bot_api.reply_message(event.reply_token,
        StickerSendMessage(package_id=event.message.package_id,sticker_id=event.message.sticker_id))
```


<LINE-Bot Reference>  
[line-bot-sdk-python](https://github.com/line/line-bot-sdk-python)  
[LINE developers](https://developers.line.me/ja/)  

## 応答LSTM ver
import sys
import random
from flask import Flask, request, abort
from myLSTM import myLSTMunit

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, PostbackEvent, TextMessage, TextSendMessage, StickerMessage, StickerSendMessage, ImageSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackTemplateAction,
)

app = Flask(__name__)
lstmunit = myLSTMunit()

line_bot_api = LineBotApi('**********')
handler = WebhookHandler('****')

@app.route("/")
def hello_world():
    return 'OK'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    
    # handle webhook body
    try:
        handler.handle(body, signature)
    
    # 最初の接続確認で引っかからないように（※重要）
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    
    # ルールベースで特定のキーワードに対応した回答をする
    if event.message.text == "おはよう":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="おはようございます"))

    # カルーセルでの選択肢付き応答
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

    ## LSTMによる応答文生成
    else:
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=lstmunit.test(event.message.text, epoch=10)))

@handler.add(MessageEvent, message=StickerMessage)
def handle_message(event):
    print(event.message)
    # print(event.message.package_id)
    # print(event.message.sticker_id)
    line_bot_api.reply_message(event.reply_token, 
        StickerSendMessage(package_id=event.message.package_id,sticker_id=event.message.sticker_id))


@handler.add(PostbackEvent)
def handle_message(event):
    event_data = event.postback.data
    if event_data == 'action=click&itemid=1':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="こんにちは"))

    elif event_data == 'action=click&itemid=2':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Hello!!"))



if __name__ == "__main__":
    # print(lstmunit.test("こんにちは", epoch=10))
    app.run()
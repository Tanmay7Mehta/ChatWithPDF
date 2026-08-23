css = '''
<style>
.chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
        background-color: #2b313e
}
.chat-message.bot {
        background-color: #475063
}
.chat-message .avatar {
        width: 15%;
}
.chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
}
.chat-message .message {
        width: 85%;
        padding: 0 1.5rem;
        color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
        <div class="avatar">
                <img src="https://i.postimg.cc/yYV0sKvW/Screenshot-2026-08-23-at-10-38-42-AM.jpg">
        </div>
        <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
        <div class="avatar">
                <img src="https://i.postimg.cc/yYV0sKvW/Screenshot-2026-08-23-at-10-38-42-AM.jpg">
        </div>
        <div class="message">{{MSG}}</div>
</div>
'''
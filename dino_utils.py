from modules import DinoV2

dino = DinoV2()

pic1 = 'data/final_submissions/17/1.jpg'
pic2 = 'data/ai/17A_11.png'
origin = 'AI'

dino.draw_attention(pic1, pic2, save=True)
dino.draw_lines(pic1, pic2, origin, save=True)
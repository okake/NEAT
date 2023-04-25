import numpy as np

empty = 0
white = -1
black = 1
wall = 2

board_size = 8

none = 0
left = 2**0 # =1
upper_left = 2**1 # =2 
upper = 2**2 # =4
upper_right = 2**3 # =8
right = 2**4 # =16
lower_right = 2**5 # =32
lower = 2**6 # =64
lower_left = 2**7 # =128

IN_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
IN_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']

MAX_TURNS = 60

class Board:
    def __init__(self):
        self.Raw_Board = np.zeros((board_size+2,board_size+2),dtype=int)
        
        self.Raw_Board[0,:] = wall
        self.Raw_Board[:,0] = wall
        self.Raw_Board[board_size+1,:] = wall
        self.Raw_Board[:,board_size+1] = wall
        
        self.Raw_Board[4,4] = white
        self.Raw_Board[5,5] = white
        self.Raw_Board[5,4] = black
        self.Raw_Board[4,5] = black
        
        self.Turns = 0
        self.CurrentColor = black
        
        self.MovablePos = np.zeros((board_size+2,board_size+2),dtype=int)
        self.MovableDir = np.zeros((board_size+2,board_size+2), dtype=int)
        self.initMovable()
 
    
    def checkMobability(self,x,y,color):
        dir = 0
        
        if(self.Raw_Board[x,y] != empty):
            return dir
        
        if(self.Raw_Board[x-1][y] == -color):
            x_tmp = x-2
            y_tmp = y
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp -= 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | left
                
        if(self.Raw_Board[x-1][y-1] == -color):
            x_tmp = x-2
            y_tmp = y-2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp -= 1
                y_tmp -= 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | upper_left
                
        if(self.Raw_Board[x][y-1] == -color):
            x_tmp = x
            y_tmp = y-2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                y_tmp -= 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | upper
                
        if(self.Raw_Board[x+1][y-1] == -color):
            x_tmp = x+2
            y_tmp = y-2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp += 1
                y_tmp -= 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | upper_right
                
        if(self.Raw_Board[x+1][y] == -color):
            x_tmp = x+2
            y_tmp = y
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp += 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | right
                
        if(self.Raw_Board[x+1][y+1] == -color):
            x_tmp = x+2
            y_tmp = y+2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp += 1
                y_tmp += 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | lower_right
                
        if(self.Raw_Board[x][y+1] == -color):
            x_tmp = x
            y_tmp = y+2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                y_tmp -= 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | lower
                
        if(self.Raw_Board[x-1][y+1] == -color):
            x_tmp = x-2
            y_tmp = y+2
            
            while(self.Raw_Board[x_tmp,y_tmp] == -color):
                x_tmp -= 1
                y_tmp += 1
            if(self.Raw_Board[x_tmp,y_tmp] == color):
                dir = dir | lower_left
                
        return dir
        
    
    def flipDiscs(self,x,y):
        self.Raw_Board[x,y] = self.CurrentColor
        
        dir = self.MovableDir[x,y]
        
        if(dir & left):
            x_tmp = x-1
            while(self.Raw_Board[x_tmp, y] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y] = self.CurrentColor
                x_tmp -= 1
                
        if(dir & upper_left):
            x_tmp = x-1
            y_tmp = y-1
            while(self.Raw_Board[x_tmp, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y_tmp] = self.CurrentColor
                x_tmp -= 1
                y_tmp -= 1
                
        if(dir & upper):
            y_tmp = y-1
            while(self.Raw_Board[x, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x, y_tmp] = self.CurrentColor
                y_tmp -= 1
                
        if(dir & upper_right):
            x_tmp = x+1
            y_tmp = y-1
            while(self.Raw_Board[x_tmp, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y_tmp] = self.CurrentColor
                x_tmp += 1
                y_tmp -= 1
                
        if(dir & right):
            x_tmp = x+1
            while(self.Raw_Board[x_tmp, y] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y] = self.CurrentColor
                x_tmp += 1
                
        if(dir & lower_right):
            x_tmp = x+1
            y_tmp = y+1
            while(self.Raw_Board[x_tmp, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y_tmp] = self.CurrentColor
                x_tmp += 1
                y_tmp += 1
                
        if(dir & lower):
            y_tmp = y+1
            while(self.Raw_Board[x, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x, y_tmp] = self.CurrentColor
                y_tmp += 1
                
        if(dir & lower_left):
            x_tmp -= 1
            y_tmp += 1
            while(self.Raw_Board[x_tmp, y_tmp] == -self.CurrentColor):
                self.Raw_Board[x_tmp, y_tmp] = self.CurrentColor
                x_tmp -= 1
                y_tmp += 1
        
    
    def move(self,x,y):
        if x < 1 or board_size < x:
            return False
        if y < 1 or board_size < y:
            return False
        if self.MovablePos[x,y] == 0:
            return False
        
        self.flipDiscs(x,y)
        self.Turns += 1
        self.CurrentColor *= -1
        self.initMovable()
        return True
        
        
    def initMovable(self):
        self.MovablePos[:,:] = False
        
        for x in range(1,board_size+1):
            for y in range(1,board_size+1):
                dir = self.checkMobability(x,y,self.CurrentColor)
                self.MovableDir[x,y] = dir
                if(dir != 0):
                    self.MovablePos[x,y] = True
                    
                    
    def checkIN(self,IN):
        if not IN:
            return False
        if IN[0] in IN_ALPHABET:
            if IN[1] in IN_NUMBER:
                return True
            
            
    def isGameOver(self):
        if(self.Turns >= MAX_TURNS):
            return True
        
        if(self.MovablePos[:,:].any()):
            return False
        
        for x in range(1,board_size+1):
            for y in range(1,board_size+1):
                if self.checkMobility(x, y, -self.CurrentColor) != 0:
                    return False
                
        return True
    
    
    def skip(self):
        if(any(MovablePos[:,:])):
            return False
        
        if(isGameOver()):
            return False
        
        self.CurrentColor *= -1
        self.initMovable()
        
        return True
                    
    
    def display(self):
        print(' abcdefgh')
        for y in range(1,9):
            print(y,end="")
            raw = ""
            for x in range(1,9):
                grid = self.Raw_Board[x,y]
                if(grid == empty):
                    raw += '□'
                elif(grid == white):
                    raw += '○'
                elif(grid == black):
                    raw += '●'
            print(raw)
    
    
board = Board()
 
# 手番ループ
while True:
 
    # 盤面の表示
    board.display()
 
    # 手番の表示
    if board.CurrentColor == black:
        print('黒の番です:', end = "")
    else:
        print('白の番です:', end = "")
    IN = input()
    print()
 
    # 入力手をチェック
    if board.checkIN(IN):
        x = IN_ALPHABET.index(IN[0]) + 1
        y = IN_NUMBER.index(IN[1]) + 1
    else:
        print('正しい形式(例：f5)で入力してください')
        continue
 
    # 手を打つ
    if not board.move(x, y):
        print('そこには置けません')
        continue
 
    # 終局判定
    if board.isGameOver():
        board.display()
        print('おわり')
        break
 
    # パス
    if not board.MovablePos[:, :].any():
        board.CurrentColor *= -1
        board.initMovable()
        print('パスしました')
        print()
        continue
 
    
# ゲーム終了後の表示
print()
    
## 各色の数
count_black = np.count_nonzero(board.Raw_Board[:, :] == black)
count_white = np.count_nonzero(board.Raw_Board[:, :] == white)
    
print('黒:', count_black)
print('白:', count_white)
 
## 勝敗
dif = count_black - count_white
if dif > 0:
    print('黒の勝ち')
elif dif < 0:
    print('白の勝ち')
else:
    print('引き分け')
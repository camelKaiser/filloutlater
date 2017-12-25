import chess
import copy


globalEncoding = {'P':0, 'R':8, 'B':10, 'N':12, 'K':14, 'Q':15, 'promoW': 16, 'p':24, 'r':32, 'b':34, 'n':36, 'k':38, 'q':39, 'promoB':40}
globalEnds = {'P':7, 'R':9, 'B':11, 'N':13, 'K':14, 'Q':15, 'p':31, 'r':33, 'b':35, 'n':37, 'k':38, 'q':39}
board = chess.Board()

def turn():
	return board.turn
def reset():
	return board.reset()
def push(move):
	return board.push(move)
def push_san(moveString):
	return board.push_san(moveString)
def pop():
	return board.pop()
	
def getFeatures():
	input = []
	#feature order
	#side to move's castling rights, opponent's rights, side to move's pieces, opponent pieces
	
	wKSC = board.castling_rights & chess.BB_H1	#white kingside castle
	bKSC = board.castling_rights & chess.BB_H8	#black kingside castle
	wQSC = board.castling_rights & chess.BB_A1	#white queenside castle
	bQSC = board.castling_rights & chess.BB_A8	#black queenside castle
	pieceList = getPieceList()
	
	if board.turn: #White's move == true, Black's move == false
		input.append(int(bool(wKSC)))
		input.append(int(bool(wQSC)))
		input.append(int(bool(bKSC)))
		input.append(int(bool(bQSC)))
		for element in pieceList:
			input += element
	else:
		input.append(int(bool(bKSC)))
		input.append(int(bool(bQSC)))
		input.append(int(bool(wKSC)))
		input.append(int(bool(wQSC)))
		
		for i in range(24):
			input += pieceList[i+24]
		for i in range(24):
			input += pieceList[i]
	return input
def getPieceList():
	temporaryEncoding = copy.deepcopy(globalEncoding)
	pieceLists = []
	for i in range(48):
		pieceLists.append([0, 0, 0])
	#First 8: white pawns
	#Next 2: Rooks
	#Next 2: Bishops (light then dark)
	#Next 2: Knights
	#King
	#Queen
	#8 hypothetical spots for promotions
	#repeat for black
	
	map = board.piece_map()
	#f = open("indexes.txt", "w")
	#f.write(str(map) + "\n")
	for key in map:
		Ycoord = int(key/8)
		Xcoord = key%8
		
		index = temporaryEncoding[map[key].symbol()]
		
		#special case to track light and dark square bishop
		if map[key].symbol() == 'b' or map[key].symbol() == 'B':
			if Ycoord % 2 == 0:			#check for black square
				if Xcoord % 2  == 0:	
					index += 1
			else:
				if Xcoord % 2 == 1:
					index += 1
		else:	
			#print (str(Xcoord) + " " + str(Ycoord) + " " + str(map[key].symbol()) + " " + str(index))
			#index = temporaryEncoding[map[key].symbol()]
			temporaryEncoding[map[key].symbol()] += 1
		prescence = 1
		#check for promo
		
		if temporaryEncoding[map[key].symbol()] > globalEnds[map[key].symbol()]:
			if map[key].color:
				index = temporaryEncoding['promoW']
				temporaryEncoding['promoW'] += 1
				prescence += map[key].piece_type
			else:
				index = temporaryEncoding['promoB']
				temporaryEncoding['promoB'] += 1
				prescence += map[key].piece_type
		#print(index)
		#f.write(str(index) + "\n")
		#f.write(str(temporaryEncoding))
		#f.write("\n")
		#f.write(str(map[key].symbol()) +"\n")
		pieceLists[index] = [float(Xcoord-3.5), float(Ycoord-3.5), prescence]	
	#f.close()
	return pieceLists

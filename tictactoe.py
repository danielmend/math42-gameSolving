import numpy as np

class Board2D:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros(shape=(board_size, board_size))
        self.pieces = [-1, 1]
    
    def __str__(self):
        return str(self.board)
    
    def place(self, piece, loc):
        row, col = loc
        self.board[row, col] = piece
        
    def is_winning_(self, arr, piece):
        return all(arr == piece)
            
    def check_diag(self):
        diag = np.diagonal(self.board)
        off_diag = np.diagonal(np.fliplr(self.board))
        for arr in [diag, off_diag]:
            for piece in self.pieces:
                if self.is_winning_(arr, piece):
                    return piece

        return None
    
    def check_rows(self):
        for row_ix in range(self.board_size):
            row = self.board[row_ix, :]
            for piece in self.pieces:
                if self.is_winning_(row, piece):
                    return piece
        return None

    def check_cols(self):
        for col_ix in range(self.board_size):
            col = self.board[:, col_ix]
            for piece in self.pieces:
                if self.is_winning_(col, piece):
                    return piece
        return None

    def get_winner(self):
        checks = (self.check_diag(), self.check_rows(), self.check_cols())
        return -1 if -1 in checks else 1 if 1 in checks else None

class Board3D:
    def __init__(self, **kwargs):
        self.board_size = kwargs.get('board_size', 3)
        self.board = kwargs.get('board', np.zeros(shape=(self.board_size, self.board_size, self.board_size)))
        self.pieces = [-1, 1]
    
    def __str__(self):
        return str(self.board)
    
    def place(self, piece, loc):
        i, j, k = loc
        self.board[i, j, k] = piece
        
    def is_winning_(self, arr, piece):
        return all(arr == piece)
    
    def get_diags(self):
        return [
            np.diagonal(np.einsum('ijj->ij', self.board)),
            np.diagonal(np.fliplr(np.einsum('ijj->ij', self.board))),
            np.diagonal(np.einsum('ijj->ij', np.fliplr(self.board))),
            np.diagonal(np.fliplr(np.einsum('ijj->ij', np.fliplr(self.board))))
        ]
    
    def get_winner(self):
        checks = []
        for idx in range(self.board_size):
            checks.extend([
                Board2D(board=self.board[idx, :, :]).get_winner(),
                Board2D(board=self.board[:, idx, :]).get_winner(),
                Board2D(board=self.board[:, :, idx]).get_winner()
            ])
        diags = self.get_diags()
        for diag in diags:
            checks.extend(
                piece if self.is_winning_(diag, piece) else 0 for piece in self.pieces
            )
        return -1 if -1 in checks else 1 if 1 in checks else 0
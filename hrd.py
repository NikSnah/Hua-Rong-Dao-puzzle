import argparse
import sys
import heapq

# python3 hrd.py --algo astar --inputfile med2.txt --outputfile med2_astar.txt 

char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """
    def __init__(self, is_blank, is_2_by_2, is_single, coord_x, coord_y, orientation, type):
        """
        :param is_2_by_2: True if the piece is a 2x2 piece and False otherwise.
        :type is_2_by_2: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
        if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """
        self.is_blank = is_blank
        self.is_2_by_2 = is_2_by_2
        self.is_single = is_single
        self.type = type
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation
        self.astar_used = False
    def set_coords(self, coord_x, coord_y):
        """
        Move the piece to the new coordinates.
        :param coord: The new coordinates after moving.
        :type coord: int
        """
        self.coord_x = coord_x
        self.coord_y = coord_y
    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_2_by_2, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)
    
    def __lt__(self, other):
        """
        Less than operator to compare two Piece objects.
        For simplicity, you can compare their coordinates or type.
        """
        if self.coord_x == other.coord_x:
            return self.coord_y < other.coord_y
        return self.coord_x < other.coord_x
class Board:
    """
    Board class for setting up the playing board.
    """
    def __init__(self, height, blanks, pieces_dict, blanks_dict):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """
        self.width = 4
        self.height = height
        self.pieces_dict = pieces_dict
        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()
        self.blanks = blanks
        self.blanks_dict = blanks_dict
        self.string_array = None
        self.display()

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False
    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location
        information.
        """
        visited = set()
        for y in range (self.height):
            line = []
            for x in range(self.width):  
                piece = self.pieces_dict[(x, y)]
                if piece.is_2_by_2:
                    line.append ('1')
                    visited.add(str(piece))
                elif piece.is_single:
                    line.append(char_single)
                    visited.add(str(piece))
                elif piece.is_blank:
                    line.append('.')
                    visited.add(str(piece))
                else:
                    if piece.orientation == 'h':
                        if str(piece) not in visited:
                            line.append('<')
                            visited.add(str(piece))
                        else: 
                            line.append('>')
                    elif piece.orientation == 'v':
                        if str(piece) not in visited:
                            line.append('^')
                            visited.add(str(piece))
                        else: 
                            line.append('v')
          
            self.grid.append(line)

        return

    def display(self):
        """
        Print out the current board and save it my self.string_array for future use.
        """

        if self.string_array == None or self.string_array == '':
            final_string = ''
            for i, line in enumerate(self.grid):
                for ch in line:
                    final_string += ch
                final_string += '\n'
            self.string_array = final_string
            return final_string
        else: 
            return self.string_array
            



class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """
    def __init__(self, board, hfn, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic]
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.hfn = hfn
        self.f = f
        self.depth = depth
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        if isinstance(other, State):
            return self.board.display() == other.board.display() and self.f == other.f
        return False


    def read_from_file(filename):
        """
        Load initial board from a given file.
        :param filename: The name of the given file.
        :type filename: str
        :return: A loaded board
        :rtype: Board
        """
        puzzle_file = open(filename, "r")
        line_index = 0
        blank_pieces = []
        blank_pieces_dict = {}
        final_blank_pieces = []
        final_blank_pieces_dict = {}
        pieces = [] # not updated during successor search and state generation
        pieces_dict = {}
        final_pieces = []
        final_pieces_dict = {}
        final = False
        found_2by2 = False
        finalfound_2by2 = False
        height_ = 0
        for line in puzzle_file:
            height_ += 1
            if line == '\n':
                if not final:
                    height_ = 0
                    final = True
                    line_index = 0
                continue
            if not final: #initial board
                for x, ch in enumerate(line):
                    if ch == '^': # found vertical piece
                        piece = Piece(False, False, False, x, line_index, 'v', 'vdouble')
                        pieces.append(piece)
                        pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
                        pieces_dict[(x, line_index + 1)] =  pieces[len(pieces) - 1]
                    elif ch == '<': # found horizontal piece
                        piece = Piece(False, False, False, x, line_index, 'h', 'hdouble')
                        pieces.append(piece)
                        pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
                        pieces_dict[(x + 1, line_index)] =  pieces[len(pieces) - 1]
                    elif ch == char_single:
                        piece = Piece(False, False, True, x, line_index, None, 'single')
                        pieces.append(piece)
                        pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
                    elif ch == '1':
                        if found_2by2 == False:
                            piece = Piece(False, True, False, x, line_index, None, '2by2')
                            pieces.append(piece)
                            pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
                            pieces_dict[(x + 1 , line_index)] =  pieces[len(pieces) - 1]
                            pieces_dict[(x, line_index + 1)] =  pieces[len(pieces) - 1]
                            pieces_dict[(x + 1, line_index + 1)] =  pieces[len(pieces) - 1]
                            found_2by2 = True
                    elif ch == '.':
                        piece = Piece(True, False, False, x, line_index, None, 'blank')
                        pieces.append(piece)
                        pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
                        blank_pieces.append(piece)
                        blank_pieces_dict[(x, line_index)] =  pieces[len(pieces) - 1]
            else: #goal board
                for x, ch in enumerate(line):
                    if ch == '^': # found vertical piece
                        piece = Piece(False, False, False, x, line_index, 'v', 'vdouble')
                        final_pieces.append(piece)
                        final_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]
                        final_pieces_dict[(x, line_index + 1)] =  final_pieces[len(final_pieces) - 1]
                    elif ch == '<': # found horizontal piece
                        piece = Piece(False, False, False, x, line_index, 'h', 'hdouble')
                        final_pieces.append(piece)
                        final_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]
                        final_pieces_dict[(x + 1, line_index)] =  final_pieces[len(final_pieces) - 1]
                    elif ch == char_single:
                        piece = Piece(False, False, True, x, line_index, None, 'single')
                        final_pieces.append(piece)
                        final_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]
                    elif ch == '1':
                        if finalfound_2by2 == False:
                            piece = Piece(False, True, False, x, line_index, None, '2by2')
                            final_pieces.append(piece)
                            final_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]
                            final_pieces_dict[(x + 1 , line_index)] =  final_pieces[len(final_pieces) - 1]
                            final_pieces_dict[(x, line_index + 1)] =  final_pieces[len(final_pieces) - 1]
                            final_pieces_dict[(x + 1, line_index + 1)] =  final_pieces[len(final_pieces) - 1]
                            finalfound_2by2 = True
                    elif ch == '.':
                        piece = Piece(True, False, False, x, line_index, None, 'blank')
                        final_pieces.append(piece)
                        final_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]
                        final_blank_pieces.append(piece)
                        final_blank_pieces_dict[(x, line_index)] =  final_pieces[len(final_pieces) - 1]

            line_index += 1
    
        puzzle_file.close()
        board = Board(height_, blank_pieces, pieces_dict, blank_pieces_dict)
        goal_board = Board(height_, final_blank_pieces, final_pieces_dict, final_blank_pieces_dict) 
        return (board, goal_board)
    

class Blank_Shapes():
    
    def __init__(self, type, coord_x, coord_y,):
        self.type = type # string for single, vdouble, hdouble
        self.coord_x = coord_x #x coordinate of the top left corner of the blank piece
        self.coord_y = coord_y #y coordinate of the top left corner of the blank piece 

    def find_all_shapes_around(self, state: State, existing_shapes, final_states):
        """
        Find all the shapes around the blank piece. 
        :param state: The current state of the puzzle.
        :return: A list of Blank_Shapes objects representing the shapes around the blank piece. 
        """
        width = state.board.width
        height = state.board.height
        shapes = []
        state_dict = state.board.pieces_dict

        # Directional offsets for up, down, left, right
        directions = {
            'r': (1, 0),  # right: x+1, y
            'l': (-1, 0),  # left: x-1, y
            'u': (0, -1),  # up: x, y-1
            'd': (0, 1),  # down: x, y+1
        }

        opposite = {
            'r': 'l',
            'l': 'r',
            'u': 'd',
            'd': 'u',
        }

        # Dictionary to store the possible moves for each direction and what the piece symbolw must be for it to move, 
        # will only move into one space
        moves = {
            'r': ('<', '2'),
            'l': ('>', '2'),
            'u': ('v', '2'),
            'd': ('^', '2'),

        }

        # For each direction, check the neighboring cell
        for direction, (dx, dy) in directions.items():
            nx, ny = self.coord_x + dx, self.coord_y + dy

            # Skip checks if the next coordinates are out of bounds
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue

            neighbor = state.board.grid[ny][nx]  # Check the neighbor cell

            if neighbor == '.':
                # Create a new blank in this direction
                create_new_blank(state, shapes, nx, ny, direction, existing_shapes, final_states)
            elif neighbor in moves[direction]:  # Check if the piece is a proper piece for a single space move 
                piece = state_dict[(nx, ny)]  # Retrieve the piece by its top-left coordinate
                move_piece(state, self, piece, opposite[direction], final_states) # Move the piece in the opposite direction
            
        return shapes
    
    def __repr__(self):
        return '{} {} {}'.format(self.coord_x, self.coord_y,  self.type)
    
def create_new_blank(state, shapes, coord_x, coord_y, pos, existing_shapes, final_states):

    def add_new_blank(blank_shape):
        """Helper function to add a blank shape if not already in the set."""
        if str(blank_shape) not in existing_shapes:
            shapes.append(blank_shape)
            existing_shapes.add(str(blank_shape))
            return True
        return False

    # Create a new single blank
    new_blank = Blank_Shapes('single', coord_x, coord_y)
    if add_new_blank(new_blank):
        # Create a double blank (vertical or horizontal) based on the position
        if pos in ['u', 'd']:  # Vertical blank
            new_double_blank = Blank_Shapes('vdouble', coord_x, coord_y if pos == 'u' else coord_y - 1)
            if add_new_blank(new_double_blank): 
                # Now check left and right of the vdouble for possible moves
                check_horizontal_moves_for_vdouble(state, new_double_blank, final_states)
                # Recursively check surrounding blanks
                shapes.extend(new_blank.find_all_shapes_around(state, existing_shapes, final_states))
        elif pos in ['l', 'r']:  # Horizontal blank
            new_double_blank = Blank_Shapes('hdouble', coord_x if pos == 'l' else coord_x - 1, coord_y)
            if add_new_blank(new_double_blank):
                # Now check above and below the hdouble for possible moves
                check_vertical_moves_for_hdouble(state, new_double_blank, final_states)
                # Recursively check surrounding blanks
                shapes.extend(new_blank.find_all_shapes_around(state, existing_shapes, final_states))
    return 

def check_horizontal_moves_for_vdouble(state, vdouble, final_states):
    """
    Check if there are any pieces to the left or right of the vertical double blank that can move into it.
    :param state: The current state of the puzzle.
    :param vdouble: The vertical double blank object.
    :param final_states: List of final states to track new valid states.
    """
    coord_x, coord_y = vdouble.coord_x, vdouble.coord_y
    board = state.board.grid
    state_dict = state.board.pieces_dict

    # Check the piece directly to the left of the vdouble
    if coord_x - 1 >= 0:  # Make sure it's within bounds
        piece_left = board[coord_y][coord_x - 1]
        if piece_left == '^':  # If it's a movable piece
            move_piece(state, vdouble, state_dict[(coord_x - 1, coord_y )], 'r', final_states)
        elif piece_left == '1':
             if board[coord_y + 1][coord_x - 1] == '1':
                    move_piece(state, vdouble, state_dict[(coord_x - 1, coord_y)], 'r', final_states)

    # Check the piece directly to the right of the vdouble
    if coord_x + 1 < len(board[0]):  # Make sure it's within bounds
        piece_right = board[coord_y][coord_x + 1]
        if piece_right == '^':
            move_piece(state, vdouble, state_dict[(coord_x + 1, coord_y)], 'l', final_states)
        elif piece_right == '1':
            if board[coord_y + 1][coord_x + 1] == '1':
                move_piece(state, vdouble, state_dict[(coord_x + 1, coord_y)], 'l', final_states)
    return

def check_vertical_moves_for_hdouble(state, hdouble, final_states):
    """
    Check if there are any pieces above or below the horizontal double blank that can move into it.
    :param state: The current state of the puzzle.
    :param hdouble: The horizontal double blank object.
    :param final_states: List of final states to track new valid states.
    """
    coord_x, coord_y = hdouble.coord_x, hdouble.coord_y
    board = state.board.grid
    state_dict = state.board.pieces_dict

    # Check the piece directly above the hdouble
    if coord_y - 1 >= 0:  # Make sure it's within bounds
        piece_above = board[coord_y - 1][coord_x]
        if piece_above == '<':  # If it's a movable piece
            move_piece(state, hdouble, state_dict[(coord_x, coord_y -1 )], 'd', final_states)
        elif piece_above == '1':
            if board[coord_y - 1][coord_x + 1] == '1':
                move_piece(state, hdouble, state_dict[(coord_x, coord_y - 1)], 'd', final_states)

    # Check the piece directly below the hdouble
    if coord_y + 1 < len(board):  # Make sure it's within bounds
        piece_below = board[coord_y + 1][coord_x]
        if piece_below == '<':
            move_piece(state, hdouble, state_dict[(coord_x, coord_y + 1)], 'u', final_states)
        elif piece_below == '1':
            if board[coord_y + 1][coord_x + 1] == '1':
                move_piece(state, hdouble, state_dict[(coord_x, coord_y + 1)], 'u', final_states)
    return

def get_successors(state: State) -> list[State]:
    """
    Get the successors of the current state.

    :param state: The current state of the puzzle.
    :return: A list of State objects representing the possible moves.
    """

    blanks_dict = state.board.blanks_dict
    #look at the current state board and check for possible moves off the blank pieces 

    curr_blank_shapes = []
    existing_shapes = set()
    curr_blank = None
    final_states = []



    #finds all the blanks and their possible shapes for a move, single or double
    for key in blanks_dict.keys(): 
        blank = blanks_dict[key]
        curr_blank = Blank_Shapes('single', blank.coord_x, blank.coord_y)

        if str(curr_blank) not in existing_shapes:
            curr_blank_shapes.append(curr_blank)
            existing_shapes.add(str(curr_blank))
            curr_blank_shapes.extend(curr_blank.find_all_shapes_around(state, existing_shapes, final_states)) 

    
    return final_states

  

def move_piece(state: State, blank_space: Blank_Shapes, piece: Piece, direction: str, final_states) -> State:
    """
    Move the piece in the given direction and return a new State object.

    :param state: The current state of the puzzle.
    :param piece: The piece to move.
    :param direction: The direction to move the piece.
    :return: A new State object with the piece moved.

    """

    shape_move = {
        'u': (0, -1),  # up: x, y-1
        'd': (0, 1),  # down: x, y+1
        'l': (-1, 0),  # left: x-1, y
        'r': (1, 0),  # right: x+1, y
    }

    # Copy the current board and pieces
    height = state.board.height
    new_pieces_dict = state.board.pieces_dict.copy()
    blanks = state.board.blanks.copy() #would need to update 
    new_blanks_dict = state.board.blanks_dict.copy()

    # Create a new board with the new piece moved
    # Get the movement vector
    dx, dy = shape_move[direction]

    new_blank_x, new_blank_y, new_blank_x2, new_blank_y2 = None, None, None, None
    new_piece_x, new_piece_y, new_piece_x2, new_piece_y2 = None, None, None, None
    new_piece_x3, new_piece_y3, new_piece_x4, new_piece_y4 = None, None, None, None

    # Helper to create pieces and move them in the dictionary 
    def create_piece_move(): 
        if new_piece_x != None and new_piece_y != None:
            change_piece_dict(new_piece_x, new_piece_y)

        if new_piece_x2 != None and new_piece_y2 != None:
            change_piece_dict(new_piece_x2, new_piece_y2)

        if new_piece_x3 != None and new_piece_y3 != None:
            change_piece_dict(new_piece_x3, new_piece_y3)

        if new_piece_x4 != None and new_piece_y4 != None:
            change_piece_dict(new_piece_x4, new_piece_y4)

        if new_blank_x != None and new_blank_y != None:
            change_piece_dict(new_blank_x, new_blank_y, True)

        if new_blank_x2 != None and new_blank_y2 != None:
            change_piece_dict(new_blank_x2, new_blank_y2, True, True)


    def change_piece_dict(new_x, new_y, is_blank=False, is_blank2=False):
        if is_blank:
            new_piece = Piece(True, False, False, new_x, new_y, None, 'blank')

            if is_blank2:
                if blank_space.type == 'vdouble':
                    new_blanks_dict.pop((blank_space.coord_x, blank_space.coord_y + 1))
                elif blank_space.type == 'hdouble':
                    new_blanks_dict.pop((blank_space.coord_x + 1, blank_space.coord_y))
            else:
                new_blanks_dict.pop((blank_space.coord_x, blank_space.coord_y))
            new_blanks_dict[(new_x, new_y)] = new_piece
        else:
            new_piece = Piece(piece.is_blank, piece.is_2_by_2, piece.is_single, new_piece_x, new_piece_y, piece.orientation, piece.type)
        new_pieces_dict[(new_x, new_y)] = new_piece


    # Update the coordinates of the piece (where the space is)
    new_piece_x = piece.coord_x + dx
    new_piece_y = piece.coord_y + dy

    # Handle different types of pieces
    if piece.is_single:

        # Update the blank space's coordinates (where the piece was)
        new_blank_x = piece.coord_x
        new_blank_y = piece.coord_y

    elif piece.type == 'vdouble':

        # second piece to move coordinates 
        new_piece_x2 = piece.coord_x + dx
        new_piece_y2 = piece.coord_y + 1 + dy

        # Update the blank space's coordinates (where the piece was)
        new_blank_x = piece.coord_x
        new_blank_y = piece.coord_y

        if direction == 'u':
            
            new_blank_y = new_blank_y + 1 

        elif direction == 'l' or direction == 'r':
            # Update the blank space's coordinates (where the piece was)
            new_blank_x2 = piece.coord_x
            new_blank_y2 = piece.coord_y + 1

    elif piece.type == 'hdouble':

        # second piece to move coordinates,  
        new_piece_x2 = piece.coord_x + 1 + dx
        new_piece_y2 = piece.coord_y + dy

        new_blank_y = piece.coord_y
        new_blank_x = piece.coord_x

        if direction == 'l':
            # Update the blank space's coordinates (where the piece was)
            new_blank_x = new_blank_x + 1
        
        elif direction == 'u' or direction == 'd':
            # Update the blank space's coordinates (where the piece was)
            new_blank_x2 = piece.coord_x + 1
            new_blank_y2 = piece.coord_y
        
    elif piece.type == '2by2':

        # second piece to move coordinates,  
        new_piece_x2 = piece.coord_x + 1 + dx
        new_piece_y2 = piece.coord_y + dy

        new_piece_x3 = piece.coord_x + dx
        new_piece_y3 = piece.coord_y + 1 + dy

        new_piece_x4 = piece.coord_x + 1 + dx
        new_piece_y4 = piece.coord_y + 1 + dy

        if direction == 'l' or direction == 'u':
            # Update the blank space's coordinates (where the piece was)

            if direction == 'l':
                new_blank_x = piece.coord_x + 1
                new_blank_y = piece.coord_y
            else:
                new_blank_x = piece.coord_x
                new_blank_y = piece.coord_y + 1

            new_blank_x2 = piece.coord_x + 1
            new_blank_y2 = piece.coord_y + 1


        elif direction == 'r' or direction == 'd':
            # Update the blank space's coordinates (where the piece was)
            new_blank_x = piece.coord_x
            new_blank_y = piece.coord_y

            if direction == 'r':
                new_blank_x2 = piece.coord_x
                new_blank_y2 = piece.coord_y + 1
            else: 
                new_blank_x2 = piece.coord_x + 1 
                new_blank_y2 = piece.coord_y 


    create_piece_move()

    # Create a new Board with the updated grid and pieces
    new_board = Board(state.board.height, blanks, new_pieces_dict, new_blanks_dict)

    # Create a new State object with the updated board
    heuristic = manhatten_distance(goal_state, piece)
    depth = state.depth + 1
    new_state = State(new_board, heuristic, heuristic + depth, depth, parent=state)

    # Add the new state to the final_states list
    final_states.append(new_state)

    return

def manhatten_distance( goal_state: State, piece_moved: Piece) -> int:
    """
    Calculate the manhatten distance of the current board.

    :param board: The current board of the puzzle.
    :return: The manhatten distance of the board.
    """

    goal_state_dict = goal_state.board.pieces_dict
    type = piece_moved.type 
    px = piece_moved.coord_x
    py = piece_moved.coord_y 


    # Initialize the manhatten distance
    possible_distances = []

    # find all the pieces on the goal board with their manhatten distance
    for key in goal_state_dict.keys(): 
        piece = goal_state_dict[key]

        if piece.type == type and not piece.astar_used:
            # Calculate the manhatten distance
            distance = abs(px - piece.coord_x) + abs(py - piece.coord_y)
            # Add to the heap with that distance with that peice being the value and distance the key
            heapq.heappush(possible_distances, (distance, piece))

    # Get the optimal (smallest) Manhattan distance
    if possible_distances:
        optimal = heapq.heappop(possible_distances)
        # Mark the piece in the goal state as used
        optimal[1].astar_used = True
        return optimal[0]  # Return the smallest distance
    else:
        return goal_state.board.width * goal_state.board.height # Return a large value if no matching piece is found

def a_star_search(initial_state: State, goal_state: State) -> list[State]:
    """

    Perform A* search to find the solution path from the initial state to the goal state.

    :param initial_state: The initial state of the puzzle (starting configuration).
    :param goal_board: The board configuration we are trying to reach.
    :return: A list of State objects representing the path from the start to the goal state.
    """
    goal_string = goal_state.board.display()
    visited = {}

    open_set = []
    heapq.heappush(open_set, (0, initial_state))  # Priority queue with (f(state), state)
    visited[initial_state.board.display()] = initial_state.f  

    while open_set:
        _, current_state = heapq.heappop(open_set)

        current_state_string = current_state.board.display()

        # If we have reached the goal, return the solution path
        if current_state_string == goal_string:
            return reconstruct_path(current_state)

            # Generate successors
        for successor in get_successors(current_state):
            successor_string = successor.board.display()
            f_cost = successor.f

            if successor_string not in visited:
                visited[successor_string] = f_cost
                heapq.heappush(open_set, (f_cost, successor))
            else: 

                if visited[successor_string] > f_cost: 
                    visited[successor_string] = f_cost
                    heapq.heappush(open_set, (f_cost, successor))

    return None  # No solution found

# helper to build the path from the initial state to the goal state
def reconstruct_path(state: State):
    path = []
    while state:
        path.append(state)
        state = state.parent
    return path[::-1]  # Return reversed path from initial to goal

#return a list of states in the order to reach the goal state
def dfs_search (initial_state: State, goal_state: State) -> list[State]: 
    """
    Perform DFS to find the solution path from the initial state to the goal state.

    :param initial_state: The initial state of the puzzle (starting configuration).
    :param goal_board: The board configuration we are trying to reach.
    :return: A list of State objects representing the path from the start to the goal state.
    """
    stack = [[initial_state, []]]
    visited = set()

    goal_str_grid = goal_state.board.display()

    while stack:
        current_state, path = stack.pop()
        current_str_grid = current_state.board.display() 

        if current_str_grid  not in visited:
            visited.add(current_str_grid )
            if current_str_grid  == goal_str_grid:
                return path + [current_state]
            for successor in get_successors(current_state):
                stack.append((successor, path + [current_state]))
    return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()
    # read the board from the file

    board, goal_board =  State.read_from_file(args.inputfile)

    # Create a initial State object
    f_score = board.width * board.height * 5
    initial_state = State(board, f_score, f_score, 0, None)


    # Create a goal State object
    goal_state = State(goal_board, 0, 0, 0, None)

    # Call the search algorithm based on the input argument
    path = None
    if args.algo == 'astar':
        # Call the A* search algorithm
        path = a_star_search(initial_state, goal_state)
    elif args.algo == 'dfs':
        # Call the DFS search algorithm
        path = dfs_search(initial_state, goal_state)

    if path == None: 
        with open(args.outputfile, 'w') as sys.stdout:
            print('No solution')
    else:
        with open(args.outputfile, 'w') as sys.stdout:
            for state in path :
                print(state.board.display())
                    
    
    #An example of how to write solutions to the outputfile. (This is not a correct solution, of course).
    #with open(args.outputfile, 'w') as sys.stdout:
    # board.display()
    # print("")
    # goal_board.display()



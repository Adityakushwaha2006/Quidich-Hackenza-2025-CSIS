import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Game settings
width, height = 1280, 640
player_width, player_height = 50, 50
player_pos = [width // 2 - player_width // 2, height - player_height - 10]

# Enemy settings
enemy_size_range = (30, 120)
enemy_list = []
enemy_speed = 10

# Initialize score and lives
score = 0
lives = 3
game_over = False

# Function to create a random enemy
def create_enemy():
    x = random.randint(0, width - enemy_size_range[1])  # Random x position
    size = random.randint(*enemy_size_range)  # Random enemy size
    speed = random.randint(10, 25)  # Random speed for each enemy
    return [x, 0, size, speed]

# Move enemies down
def move_enemies(enemy_list):
    global score
    for enemy in enemy_list[:]:  # Iterate over a copy of the list
        enemy[1] += enemy[3]  # Move down based on enemy speed
        if enemy[1] > height:  # If enemy goes off-screen
            enemy_list.remove(enemy)
            score += 1  # Increment score for each enemy that goes off-screen

# Check for collisions
def check_collision(player_pos, enemy_list):
    px, py = player_pos
    for enemy in enemy_list:
        ex, ey, es, _ = enemy  # Decompose enemy properties
        # Check if player's bounding box collides with enemy's bounding box
        if (px < ex + es and px + player_width > ex and
            py < ey + es and py + player_height > ey):
            return True
    return False

# Function to reset the game after "Game Over"
def reset_game():
    global score, lives, enemy_list, player_pos, game_over
    score = 0
    lives = 3
    game_over = False
    enemy_list = []
    player_pos = [width // 2 - player_width // 2, height - player_height - 10]

# Initialize webcam
cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not game_over:
        # Process the frame with MediaPipe
        result = hands.process(rgb_frame)

        # Get coordinates of the index finger tip (landmark 8)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_x = int(index_finger_tip.x * width)
                hand_y = int(index_finger_tip.y * height)

                # Move player based on hand movement
                player_pos[0] = np.clip(hand_x - player_width // 2, 0, width - player_width)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Add new enemies randomly
        if random.random() < 0.02:
            enemy_list.append(create_enemy())

        # Move enemies
        move_enemies(enemy_list)

        # Check for collision
        if check_collision(player_pos, enemy_list):
            lives -= 1
            if lives == 0:
                game_over = True
            else:
                # Remove all current enemies and reset position
                enemy_list = []

        # Draw game elements
        # Draw player
        cv2.rectangle(frame, (player_pos[0], player_pos[1]), (player_pos[0] + player_width, player_pos[1] + player_height), (0, 255, 0), -1)

        # Draw enemies
        for enemy in enemy_list:
            cv2.rectangle(frame, (enemy[0], enemy[1]), (enemy[0] + enemy[2], enemy[1] + enemy[2]), (0, 0, 255), -1)

        # Draw a black background for the score and lives text
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)

        # Display score and lives on the frame with white text and black background
        cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Lives: {lives}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        # Draw a black background for the game over screen
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 0), -1)
        
        # Show game over screen and final score
        cv2.putText(frame, "GAME OVER", (width // 2 - 200, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(frame, f"Final Score: {score}", (width // 2 - 150, height // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.putText(frame, "Press 'r' to Restart or 'q' to Quit", (width // 2 - 300, height // 2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Display score on the frame
    cv2.imshow("Object Dodging Game", frame)

    # Quit, Restarting the Game
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if game_over and key == ord('r'):
        reset_game()

cap.release()
cv2.destroyAllWindows()
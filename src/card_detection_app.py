import time
import threading
import queue
import cv2

from manager.card_manager import CardManager
from manager.detection_manager import DetectionManager
from manager.player_manager import PlayerManager
from manager.round_manager import RoundManager
from manager.video_capture_manager import VideoCaptureManager
from manager.yolo_model_manager import YOLOModelManager


class CardDetectionApp:
    frame_skip = 5  # Process every nth frame to reduce workload

    def __init__(self, video_file=r"C:\Users\macma\PycharmProjects\yolo-detection-app\videos\2024-11-01 11-34-21.mkv"):
        self.video_capture_manager = VideoCaptureManager(video_file)
        self.yolo_model_manager = YOLOModelManager()
        self.player_manager = PlayerManager(self)
        self.card_manager = CardManager()
        self.round_manager = RoundManager(self)
        self.detection_manager = DetectionManager(self)
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=10)  # Shared queue for frames between threads
        self.stop_event = threading.Event()  # Event to signal threads to stop
        self.frame_count = 0
        self.player_detected = False
        self.cards_detected = True

        self.round_manager.setup_round_robin()
        print(f"Round {self.round_manager.round_number} starting.")
        print(f"Matchups: {self.round_manager.current_matchups}")

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time}")
        return elapsed_time

    def check_round_end(self):
        if (self.round_manager.round_number <= self.round_manager.first_phase_rounds and len(
                self.card_manager.cards_first_set) == 4 and self.player_detected and not self.cards_detected):
            self.round_manager.end_round()
        elif (self.round_manager.round_number > self.round_manager.first_phase_rounds and len(
                self.card_manager.cards_second_set) == 4 and self.player_detected and not self.cards_detected):
            self.round_manager.end_round()
        if (
                self.round_manager.round_number > self.round_manager.first_phase_rounds and self.player_detected and not self.cards_detected and len(
            self.card_manager.cards_second_set) < 4):
            self.card_manager.duplicate_cards()

    def process_card_detection(self, detected_cards):
        self.card_manager.detect_card_played(detected_cards, self.player_manager.players_first_set)
        self.check_round_end()

    def match_and_record_players_cards(self):
        matched_players_cards = []

        for players_set in [self.player_manager.players_first_set, self.player_manager.players_second_set]:
            for player, player_time in players_set:
                match = next(
                    (card for card_set in [self.card_manager.cards_first_set, self.card_manager.cards_second_set] for
                     card in card_set if self.is_match(player, card, matched_players_cards)), None)
                if match:
                    matched_players_cards.append((player, player_time, match))

        return matched_players_cards

    @staticmethod
    def is_match(player, card, matched_players_cards):
        return player[0].lower() == card[-1].lower() and (player, card) not in matched_players_cards

    def frame_capture_thread(self):
        """Thread for capturing frames from the video."""
        while not self.stop_event.is_set():
            ret, frame = self.video_capture_manager.read_frame()
            if not ret:
                self.stop_event.set()
                break
            # Add frame to the queue for processing
            try:
                self.frame_queue.put(frame, timeout=1)  # Block if queue is full
            except queue.Full:
                continue  # Drop frames if queue is full

    def frame_processing_thread(self):
        """Thread for processing frames."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)  # Wait for a frame from the queue

                # Count the frame and potentially skip
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                self.detection_manager.detect_and_process_qrcodes(frame)
                detect_result = self.yolo_model_manager.detect_objects(frame)
                detect_image = detect_result[0].plot()
                detect_players = DetectionManager.detect_players(frame)
                if detect_players:
                    self.player_detected = True
                detected_cards_indices = detect_result[0].boxes.cls.tolist()
                detected_cards = [detect_result[0].names[i] for i in detected_cards_indices]
                self.cards_detected = bool(detected_cards)
                self.detection_manager.detect_and_process_cards(frame)
                cv2.imshow('Card Detection', detect_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()

            except queue.Empty:
                continue  # Wait for the next frame

    def run(self):
        capture_thread = threading.Thread(target=self.frame_capture_thread)
        processing_thread = threading.Thread(target=self.frame_processing_thread)

        capture_thread.start()
        processing_thread.start()

        capture_thread.join()  # Wait for capture thread to finish
        self.stop_event.set()  # Signal processing thread to stop
        processing_thread.join()  # Wait for processing thread to finish

        self.video_capture_manager.release()
        cv2.destroyAllWindows()
        self.round_manager.write_round_data_to_csv()


if __name__ == "__main__":
    app = CardDetectionApp()
    app.run()
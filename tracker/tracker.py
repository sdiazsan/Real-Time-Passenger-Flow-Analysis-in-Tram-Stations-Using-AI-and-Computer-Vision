import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.lost_frames = {}  # Dictionary to track lost frames
        self.last_positions = {}  # Register the last position of IDs

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 75:  # Adjusted distance threshold
                    self.center_points[id] = (cx, cy)
                    self.last_positions[id] = (x, y, w, h)  # Update last position
                    objects_bbs_ids.append([x, y, w, h, id])
                    self.lost_frames[id] = 0
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.last_positions[self.id_count] = (x, y, w, h)  # Save new position
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.lost_frames[self.id_count] = 0
                self.id_count += 1

        # Increment lost frames and remove IDs if necessary
        for id in list(self.lost_frames.keys()):
            self.lost_frames[id] += 1
            if self.lost_frames[id] > 30:  # Increased tolerance for lost frames
                del self.center_points[id]
                del self.lost_frames[id]

        return objects_bbs_ids


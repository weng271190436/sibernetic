"""
A PyVista based viewer/player for saved Sibernetic simulations

Loads in the generated position_buffer.txt file

"""

import pyvista as pv
import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum

last_meshes = {}

replay_speed = 0.05  # seconds between frames
replaying = False

all_points = []
all_point_types = []

plotter = None
offset3d_ = (0, 0, 0)
slider = None

show_boundary = False
max_time = None

verbose = False


class State(Enum):
    PAUSED = "Paused"
    RUNNING = "Running"


class ReplayController:
    slider_view = None

    def __init__(self, times=None):
        self.times = list(times)
        self.state = State.PAUSED
        self.current_time_index = 0

    def play(self, should_play):
        if should_play:
            print(" > Starting replay playback.")

            if self.current_time_index == len(self.times)-1:
                print(" > Replay at end of time, resetting to start.")
                self.set_to_time(0)

            self.state = State.RUNNING
            while self.state == State.RUNNING and self.current_time_index + 1 < len(
                self.times
            ):
                self.current_time_index += 1
                print(f" > Advancing to time index: {self.current_time_index}")
                self.render_all()
                time.sleep(replay_speed)
            print(" > Replay playback finished or paused.")
            self.state = State.PAUSED

        else:
            print(" > Pausing replay playback.")
            self.state = State.PAUSED

    def step_forward(self):
        if self.state == State.RUNNING:
            self.state = State.PAUSED

        if self.current_time_index + 1 >= len(self.times):
            print(" > Replay at end of time, cannot step forward.")
            return

        self.current_time_index += 1
        print(f" > Stepping forward one time step to index: {self.current_time_index}")
        self.render_all()

    def step_backward(self):
        if self.state == State.RUNNING:
            self.state = State.PAUSED

        if self.current_time_index == 0:
            print(" > Replay at start of time, cannot step backward.")
            return

        self.current_time_index -= 1
        print(f" > Stepping backward one time step to index: {self.current_time_index}")
        self.render_all()

    def set_to_time(self, time_value):
        if time_value == 0:
            closest_index = 0
            closest_time = 0
        else:
            if time_value in self.times:
                closest_index = self.times.index(time_value)
                closest_time = self.times[closest_index]
                print(
                    f" > .Finding closest time to {time_value}, got index: {closest_index}"
                )
            else:
                closest = min(self.times, key=lambda x: abs(x - time_value))
                closest_index = self.times.index(closest)
                print(
                    f" > Finding closest time to {time_value}, got index: {closest_index}"
                )
                closest_time = self.times[closest_index]

        self.current_time_index = closest_index

        print(
            " > Replay requested to be set to: %s; being set to time value %s (index: %d)."
            % (time_value, self.current_time_index, closest_time)
        )
        self.state = State.PAUSED
        self.render_all()

    def render_all(self):
        print(
            " > Rendering replay at time index: %d (time: %s)"
            % (self.current_time_index, self.times[self.current_time_index])
        )
        if self.slider_view is not None:
            self.slider_view.GetSliderRepresentation().SetValue(
                self.times[self.current_time_index]
            )
        create_mesh(self.current_time_index)
        plotter.render()
        plotter.update()

    def get_state(self):
        return f" > Replay state: {self.state}, current time index: {self.current_time_index}, so time is {self.times[self.current_time_index]} of max time {self.times[-1]} ({len(self.times)} time points)."


replay_controller = None


def get_color_info_for_type(type_):
    """
    Get color, info string and point size for a given point type
    returns: color, info, size
    """

    if type_ == 1.1:
        return "cyan", "liquid 1", 2
    elif type_ == 1.2:
        return "red", "liquid 2", 10

    elif type_ == 2.1:
        return "pink", "elastic 1", 5
    elif type_ == 2.2:
        return "#FF0000", "elastic 2", 5
    elif type_ >= 2.3 and type_ < 2.4:
        return "lightyellow", "elastic variable", 5
    elif type_ > 2 and type_ < 3:
        return "#00cc00", "elastic variable", 7

    elif type_ == 3:
        return "grey", "boundary 0", 3
    elif type_ == 3.1:
        return "black", "boundary 1", 7
    else:
        return "orange", "unknown", 5


def add_sibernetic_model(
    pl,
    position_file="Sibernetic/position_buffer.txt",
    report_file=None,
    swap_y_z=False,
    offset3d=(0, 0, 0),
    include_boundary=False,
):
    global \
        all_points, \
        all_point_types, \
        last_meshes, \
        plotter, \
        offset3d_, \
        slider, \
        show_boundary, \
        max_time, \
        replay_controller

    offset3d_ = offset3d
    plotter = pl
    show_boundary = include_boundary

    points = {}
    types = []

    line_count = 0
    pcount = 0
    time_count = 0
    logStep = None

    dt = None

    report_data = None
    count_point_types = {}

    if report_file is not None:
        sim_dir = os.path.dirname(os.path.abspath(report_file))
        report_data = json.load(open(report_file, "r"))
        print(report_data)
        position_file = os.path.join(sim_dir, "position_buffer.txt")
        dt = float(report_data.get("dt").split(" ")[0])
        duration = float(report_data.get("duration").split(" ")[0])
        log_step = int(report_data.get("logstep"))

        max_time = duration
        neuron_time_points = np.linspace(0, duration, int(duration / dt) + 1)

        sibernetic_time_points = np.linspace(
            0, duration, int((duration / dt) / log_step)
        )
        replay_controller = ReplayController(times=sibernetic_time_points)

        print(
            "Simulation dt: %s ms, duration: %s ms, times simulated (%i): %s; sibernetic logged times (%i): %s"
            % (
                dt,
                duration,
                len(neuron_time_points),
                neuron_time_points,
                len(sibernetic_time_points),
                sibernetic_time_points,
            )
        )

        if "worm" in report_data["configuration"]:
            muscle_activation_file = os.path.join(
                sim_dir, "muscles_activity_buffer.txt"
            )
            print("Loading muscle activation file from: %s" % muscle_activation_file)
            musc_dat = np.loadtxt(muscle_activation_file, delimiter="\t").T
            print(musc_dat)
            print(musc_dat.shape)
            # plt.imshow(musc_dat, interpolation="none", aspect="auto", cmap="YlOrRd")

            f, ax = plt.subplots(tight_layout=True)
            ax.imshow(musc_dat, interpolation="none", aspect="auto", cmap="YlOrRd")

            num_ticks = 5
            ax.set_xticks(np.linspace(0, musc_dat.shape[1], num_ticks))
            ax.set_xticklabels(np.linspace(0, duration, num_ticks))
            # quit()

            # ax.set_ylim([-1, 1])
            ax.set_xlabel("Time (ms)")
            _ = ax.set_ylabel("Muscle")

            h_chart = pv.ChartMPL(f, size=(0.35, 0.35), loc=(0.02, 0.06))
            h_chart.title = None
            h_chart.border_color = "white"
            h_chart.show_title = False
            h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
            pl.add_chart(
                h_chart,
            )

    first_pass_complete = False

    for line in open(position_file):
        ws = line.split()
        # print(ws)
        if line_count == 6:
            numOfElasticP = int(ws[0])
        if line_count == 7:
            numOfLiquidP = int(ws[0])
        if line_count == 8:
            numOfBoundaryP = int(ws[0])
        if line_count == 9:
            timeStep = float(ws[0])  # noqa: F841
        if line_count == 10:
            logStep = int(ws[0])

        if len(ws) == 4:
            type_ = float(ws[3])
            if type_ not in points:
                points[type_] = []

            if not first_pass_complete:
                if type_ not in count_point_types:
                    count_point_types[type_] = 0
                count_point_types[type_] += 1

            if swap_y_z:
                points[type_].append([float(ws[1]), 1 * float(ws[0]), float(ws[2])])
            else:
                points[type_].append([float(ws[0]), float(ws[1]), float(ws[2])])

            types.append(type_)

        if logStep is not None:
            pcount += 1

            if pcount == numOfBoundaryP + numOfElasticP + numOfLiquidP:
                first_pass_complete = True
                print(
                    "End of one batch of %i total points (%i types), at line %i, time: %i"
                    % (pcount, len(points), line_count, time_count)
                )
                all_points.append(points)
                all_point_types.append(types)

                points = {}
                types = []
                pcount = 0
                numOfBoundaryP = 0

                time_count += 1

        line_count += 1

    print(
        "Loaded positions with %i elastic, %i liquid and %i boundary points (%i total), over %i lines"
        % (
            numOfElasticP,
            numOfLiquidP,
            numOfBoundaryP,
            numOfElasticP + numOfLiquidP + numOfBoundaryP,
            line_count,
        )
    )

    print("Num of time points found: %i" % len(all_points))

    if replay_controller is None:
        time_points = np.arange(len(all_points))
        replay_controller = ReplayController(times=time_points)

    print("Count of point types found: %s" % dict(sorted(count_point_types.items())))

    create_mesh(0)

    slider_text = "Time point"

    if max_time is None:
        max_time = len(all_points) - 1
    else:
        slider_text = "Time (ms)"

    slider = pl.add_slider_widget(
        slider_updated, rng=[0, max_time], value=0, title=slider_text, style="modern"
    )

    replay_controller.slider_view = slider

    button_height = 10
    txt_offset = 8
    txt_voffset = 12

    b1 = 10
    pl.add_checkbox_button_widget(
        back_checkbox_pressed,
        value=False,
        position=(b1, button_height),
        color_on="lightgrey",
        color_off="lightgrey",
    )
    pl.add_text(
        "<|",
        position=(b1 + txt_offset, button_height + txt_voffset),
        font_size=12,
        color="black",
    )
    b2 = 80
    pl.add_checkbox_button_widget(
        play_checkbox_pressed,
        value=False,
        position=(b2, button_height),
        color_on="lightgrey",
        color_off="darkgrey",
    )
    pl.add_text(
        " >",
        position=(b2 + txt_offset, button_height + txt_voffset),
        font_size=12,
        color="black",
    )
    b3 = 150
    pl.add_checkbox_button_widget(
        fwd_checkbox_pressed,
        value=False,
        position=(b3, button_height),
        color_on="lightgrey",
        color_off="lightgrey",
    )
    pl.add_text(
        "|>",
        position=(b3 + txt_offset, button_height + txt_voffset),
        font_size=12,
        color="black",
    )


def slider_updated(value):
    global replay_controller
    print(
        f" > Slider updated to value: {value}, replay: {replay_controller.get_state()}"
    )

    replay_controller.set_to_time(value)


def fwd_checkbox_pressed(value):
    global replay_controller
    print(f" > Fwd checkbox pressed, value: {value}")
    replay_controller.step_forward()


def play_checkbox_pressed(value):
    global replay_controller
    print(f" > Play checkbox pressed, value: {value}")
    replay_controller.play(value)


def back_checkbox_pressed(value):
    global replay_controller
    print(f" > Back checkbox pressed, value: {value}")
    replay_controller.step_backward()

'''
def play_animation(play_button_active):
    global \
        plotter, \
        last_meshes, \
        all_points, \
        all_point_types, \
        replaying, \
        slider, \
        replay_controller

    print(
        f"Animation button pressed. Button active {play_button_active}; replaying: {replaying}, slider value: {slider.GetSliderRepresentation().GetValue()}"
    )
    print(replay_controller.get_state())

    if not play_button_active:
        if not replaying:
            print("Animation already stopped - restarting")
            slider.GetSliderRepresentation().SetValue(0)
            curr_time = slider.GetSliderRepresentation().GetValue()
            replaying = True
            plotter.update()
            plotter.render()
        else:
            curr_time = slider.GetSliderRepresentation().GetValue()
            replaying = False
            print("Animation stopped at %s." % curr_time)
            return
    else:
        replaying = True
        print("Animation started.")

    if last_meshes is None:
        print("No meshes to animate. Please load a model first.")
        return

    for i in range(len(all_points)):
        if not replaying:
            break
        curr_time = slider.GetSliderRepresentation().GetValue()

        print(
            " --- Animating step %i/%i (curr_time: %s) of %i, %s"
            % (i, len(all_points), curr_time, len(all_points), play_button_active)
        )
        next_time = curr_time + 1
        slider.GetSliderRepresentation().SetValue(next_time)

        create_mesh(next_time)
        plotter.update()
        plotter.render()
        time.sleep(replay_speed)

    replaying = False'''


def create_mesh(time_index):
    global all_points, last_meshes, plotter, offset3d_, show_boundary

    if time_index >= len(all_points):
        print(
            "Index %i out of bounds for all_points with length %i"
            % (time_index, len(all_points))
        )
        return

    print(
        "   -- Creating new mesh at time point index: %s/%s"
        % (time_index, len(all_points))
    )
    curr_points_dict = all_points[time_index]

    print("      Plotting %i point types" % (len(curr_points_dict)))

    for type_, curr_points in curr_points_dict.items():
        color, info, size = get_color_info_for_type(type_)
        is_boundary = "boundary" in info
        if show_boundary is False and is_boundary:
            mx = max(curr_points)
            mn = min(curr_points)
            print(mx)
            print(mn)
            a = [mn[0], mn[1], mn[2]]
            b = [mn[0], mx[1], mn[2]]
            c = [mn[0], mx[1], mx[2]]
            d = [mn[0], mn[1], mx[2]]

            points = np.array([a, b, b, c, c, d, d, a])
            plotter.add_lines(points, color="grey", width=2)
            # quit()
            continue

        if verbose:
            print(
                "       - Plotting %i points of type '%s' (%s), color: %s, size: %i"
                % (len(curr_points), type_, info, color, size)
            )

        if len(curr_points) == 0:
            continue
        if type_ not in last_meshes:
            last_meshes[type_] = pv.PolyData(curr_points)
            last_meshes[type_].translate(offset3d_, inplace=True)

            # last_actor =
            plotter.add_mesh(
                last_meshes[type_],
                render_points_as_spheres=True,
                point_size=size,
                color=color,
            )
        else:
            if not is_boundary:
                last_meshes[type_].points = curr_points
                last_meshes[type_].translate(
                    (offset3d_[0], offset3d_[1], offset3d_[2]), inplace=True
                )
            else:
                print("Boundary points not translated")

    plotter.render()
    # time.sleep(0.1)

    return


if __name__ == "__main__":
    plotter = pv.Plotter()

    position_file = "buffers/position_buffer.txt"  # can be overwritten by arg
    report_file = None

    if not os.path.isfile(position_file):
        position_file = (
            "Sibernetic/position_buffer.txt"  # example location in Worm3DViewer repo
        )

    include_boundary = False

    if "-b" in sys.argv:
        include_boundary = True
    else:
        print("Run with -b to display boundary box")

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        if "json" in sys.argv[1]:
            position_file = None
            report_file = sys.argv[1]
        else:
            position_file = sys.argv[1]

    add_sibernetic_model(
        plotter,
        position_file,
        report_file,
        swap_y_z=True,
        include_boundary=include_boundary,
    )
    plotter.window_size = [1600, 800]

    plotter.set_background("white")
    plotter.add_axes()
    plotter.camera_position = "zx"
    plotter.camera.roll = 90
    plotter.camera.elevation = 45
    print(plotter.camera_position)

    def on_close_callback(plotter):
        global replay_controller
        print(
            f"Plotter window is closing. Performing actions now (replay: {replay_controller.get_state()})."
        )
        replay_controller.state = State.PAUSED

    if "-nogui" not in sys.argv:
        plotter.show(before_close_callback=on_close_callback, auto_close=True)
        print("Done showing")

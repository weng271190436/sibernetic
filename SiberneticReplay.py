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


last_meshes = {}

replay_speed = 0.05  # seconds between frames
replaying = False

all_points = []
all_point_types = []


plotter = None
offset_ = 0
slider = None


def get_color_info_for_type(type_):
    """
    Get color, info string and point size for a given point type
    returns: color, info, size
    """

    if type_ == 1.1:
        return "#8BEBFC", "liquid 1", 5
    elif type_ == 1.2:
        return "#3ACFF0", "liquid 2", 5
    elif type_ == 2.1:
        return "yellow", "elastic 1", 5
    elif type_ == 2.2:
        return "#FF0000", "elastic 2", 5
    elif type_ > 2 and type_ < 3:
        return "#00cc00", "elastic variable", 5
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
    offset=0,
    include_boundary=False,
):
    global all_points, all_point_types, last_meshes, plotter, offset_, slider

    offset_ = offset
    plotter = pl

    points = {}
    types = []

    line_count = 0
    pcount = 0
    time_count = 0
    logStep = None

    report_data = None
    count_point_types = {}

    if report_file is not None:
        sim_dir = os.path.dirname(os.path.abspath(report_file))
        report_data = json.load(open(report_file, "r"))
        print(report_data)
        position_file = os.path.join(sim_dir, "position_buffer.txt")

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
            # ax.set_ylim([-1, 1])
            ax.set_xlabel("Time (s)")
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
    print("Count of point types found: %s" % dict(sorted(count_point_types.items())))

    create_mesh(0)

    max_time = len(all_points) - 1

    slider = pl.add_slider_widget(
        create_mesh, rng=[0, max_time], value=0, title="Time point", style="modern"
    )

    pl.add_checkbox_button_widget(play_animation, value=False)


def play_animation(play):
    global plotter, last_meshes, all_points, all_point_types, replaying, slider
    print("Playing animation: %s" % play)

    if not play:
        replaying = False
        print("Animation stopped.")
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
            " --- Animating step %i (curr_time: %s) of %i, %s"
            % (i, curr_time, len(all_points), play)
        )
        next_time = curr_time + 1
        slider.GetSliderRepresentation().SetValue(next_time)

        create_mesh(next_time)
        plotter.update()
        plotter.render()
        time.sleep(replay_speed)


def create_mesh(step):
    step_count = step
    value = step_count
    global all_points, last_meshes, plotter, offset_, replaying

    index = int(value)
    if index >= len(all_points):
        print(
            "Index %i out of bounds for all_points with length %i"
            % (index, len(all_points))
        )
        replaying = False
        return

    print("   -- Creating new mesh at time point: %s (%s) " % (index, value))
    curr_points_dict = all_points[index]

    print("      Plotting %i point types" % (len(curr_points_dict)))

    for type_, curr_points in curr_points_dict.items():
        color, info, size = get_color_info_for_type(type_)
        is_boundary = "boundary" in info
        if include_boundary is False and is_boundary:
            continue

        print(
            "       - Plotting %i points of type '%s' (%s), color: %s, size: %i"
            % (len(curr_points), type_, info, color, size)
        )

        if len(curr_points) == 0:
            continue
        if type_ not in last_meshes:
            last_meshes[type_] = pv.PolyData(curr_points)
            last_meshes[type_].translate((0, 0, 0), inplace=True)

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
                last_meshes[type_].translate((offset_, 0, 0), inplace=True)
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
    plotter.set_background("white")
    plotter.add_axes()
    plotter.camera_position = "zx"
    plotter.camera.roll = 90
    plotter.camera.elevation = 45
    print(plotter.camera_position)

    if "-nogui" not in sys.argv:
        plotter.show()

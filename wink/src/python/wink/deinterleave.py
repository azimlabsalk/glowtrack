import click
import imageio


@click.command('deinterlace')
@click.argument('input_video')
@click.argument('output_video1')
@click.argument('output_video2')
def deinterlace(input_video, output_video1, output_video2):

    output_videos = [output_video1, output_video2]

    print('input_video: {}'.format(input_video))
    print('output_videos: {}'.format(output_videos))

    writers = []
    for output_video in output_videos:
        writer = imageio.get_writer(output_video, quality=10, fps=30)
        writers.append(writer)

    reader = imageio.get_reader(input_video)

    writer_idx = 0
    for image in reader.iter_data():
        writers[writer_idx].append_data(image)
        writer_idx = (writer_idx + 1) % len(writers)

    reader.close()
    for writer in writers:
        writer.close()


if __name__ == '__main__':
    deinterlace()

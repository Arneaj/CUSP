full_process_test: full_process_test.o preprocessing.o raycast.o magnetopause.o streamlines.o points.o read_file.o matrix.o
	g++ -pthread full_process_test.o preprocessing.o raycast.o magnetopause.o streamlines.o points.o read_file.o matrix.o -o full_process_test

raycast_test: raycast_test.o raycast.o magnetopause.o streamlines.o points.o read_file.o matrix.o
	g++ raycast_test.o raycast.o magnetopause.o streamlines.o points.o read_file.o matrix.o -o raycast_test

preprocessing_test: preprocessing_test.o  streamlines.o points.o read_file.o matrix.o
	g++ preprocessing_test.o preprocessing.o streamlines.o points.o read_file.o matrix.o -o preprocessing_test

magnetopause_test: magnetopause_test.o magnetopause.o extend_streamlines.o streamlines.o points.o read_file.o matrix.o
	g++ magnetopause_test.o magnetopause.o extend_streamlines.o streamlines.o points.o read_file.o matrix.o -o magnetopause_test

extend_streamlines_test: extend_streamlines_test.o extend_streamlines.o streamlines.o points.o read_file.o matrix.o
	g++ extend_streamlines_test.o extend_streamlines.o streamlines.o points.o read_file.o matrix.o -o extend_streamlines_test

centerline_test: centerline_test.o centerline.o streamlines.o points.o read_file.o matrix.o
	g++ centerline_test.o centerline.o streamlines.o points.o read_file.o matrix.o -o centerline_test

streamlines_benchmark: streamlines_benchmark.o streamlines.o points.o read_file.o matrix.o
	g++ streamlines_benchmark.o streamlines.o points.o read_file.o matrix.o -o streamlines_benchmark

streamlines_test: streamlines_test.o streamlines.o points.o read_file.o matrix.o
	g++ streamlines_test.o streamlines.o points.o read_file.o matrix.o -o streamlines_test

read_file_test: read_file_test.o read_file.o matrix.o
	g++ read_file_test.o read_file.o matrix.o -o read_file_test

clean:
	rm *.o

%.o: ../src_cpp/%.cpp ../headers_cpp/%.h
	g++ -pthread -O3 -Wall -c $< -o $@

%.o: ../tests_cpp/%.cpp
	g++ -O3 -Wall -c $< -o $@


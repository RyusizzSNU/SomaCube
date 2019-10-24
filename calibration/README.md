intrinsic_calibration.py : 카메라를 intrinsic calibration 하기 위해 사용
 * cam_type, m, n, size, result_dir 를 argument로 입력
 * 격자(target)을 회전시키고 이동시키면서 사진을 찍음(자동 촬영)
 * results/intrinsic/<result_dir>/mtx.pkl 에 calibration 결과
   (camera_to_image matrix)가 저장됨.

make_pos_image_pairs.py : extrinsic calibration을 위한 데이터 쌍을 얻기 위해 사용
 * cam_type, data_dir 를 argument로 입력
 * 화살표와 W/S 로 wrist의 위치를 조정, Q/E/H/K/Y/I 로 wrist의 rotation을 조정하고,
   스페이스바로 사진을 찍음.
 * 사진을 찍으면 사진이 <data_dir>/images에 저장되고, 그 시점에서 로봇의 손목 좌표가 data_dir/poses 에 저장됨.

extrinsic_calibration.py : 카메라를 extrinsic calibration하기 위해 사용
 * cam, h, w, size, data_dir 를 argument로 입력
 * cam이 static인 경우 : 손목에 붙어있는 target을 static camera로 촬영하는 경우를 상정.
                         camera_to_base matrix와 target_to_wrist matrix를 추정.

 * cam이 wrist인 경우 : 바닥에 가만히 있는 target을 wrist camera로 촬영하는 경우를 상정.
                         camera_to_wrist matrix와 target_to_base matrix를 추정.
 
 * cam이 both인 경우 : target을 wrist camera와 static camera로 동시에 촬영하는 경우를 상정.
                         static_camera_to_base matrix와 wrist_camera_to_wrist matrix를 추정.

 * intrinsic_calibration.py에서 구한 카메라 정보와, make_pos_image_pairs.py 에서 만든 데이터 쌍들을 이용해
   오브젝트 간의 좌표 변환 matrix를 추정
 * results/extrinsic/<cam>/mtx.pkl 에 결과가 저장됨.
   * cam이 static인 경우, camera_to_base matrix와 target_to_wrist matrix를 stack한 array가 저장됨.
   * cam이 wrist인 경우, camera_to_wrist matrix와 target_to_base matrix를 stack한 array가 저장됨.
   * cam이 both인 경우, static_camera_to_base matrix와 wrist_camera_to_wrist matrix를 stack한 array가 저장됨.

static_test.py : static camera에 대해 calibration한 것을 테스트하기 위해 사용
 * h, w, size, auto를 argument로 입력
 * 실행 시, static camera가 target을 보고 있어야 함(손목 등에 의해 가려지면 안됨).
 * static camera가 찍은 image에서 target을 찾고, 변환을 여러 번 시켜서 target의 base좌표계 좌표를 구함.
 * wrist를 target의 각 좌표로 이동시킴. auto == 1 인 경우 자동으로 이동하고, auto == 0 인 경우
   스페이스바를 누르면 이동함.
 
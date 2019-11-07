pose_estimation
================

- pp3.py와 grippoint.py는 DeeplabModule()을 통해 7가지 block의 종류와 위치를 구하여 로봇이 block을 잡아야 할 좌표를 계산함.

- segmentation/deepLab/deeplab manual.md에 따라 환경 구축해야 함.

- deeplab.run()에 학습시키고 싶은 image를 넣어주고 코드 전체를 실행하면 됨. ( ex)out = deeplab.run("blocks.jpg") )

- input image를 학습시켜 블록의 lable을 구분함.
OpenCV의 contourArea()로 영역의 크기를 지정해서 선택된 영역에만 boundingRect()를 쳐서 block의 angle을 구함.
(이때 angle은 절대값이 아닌 90이하의 값)

- 정확한 회전각을 구하기 위해서 OpenCV의 matchTemplate()을 이용('TM_CCOEFF_NORMED' 사용).
template 리스트는 mk_ps_tp()에서 생성됨.

- lable_to_pose에서 lable 당 가능한 pose 후보군을 설정.
boundingRect()를 통해 구한 angle을 90 간격으로 4가지 후보군 설정.

- 위에 대한 결과로 얻은 결과로 회전된 절대각을 구하여 pose1(),pose2(),pose3(),pose4(),pose5(),pose6(),pose7(),pose8()에서 잡아야 할 좌표를 계산해줌.
pose가 같더라도 block 당 잡아야할 위치가 다르기 때문에 lable에 따라 다르게 계산되도록 함.

- pp3의 출력값은 template 매칭된 결과의 대각선 좌표들이며, grippoint의 결과는 잡아야 할 좌표임.







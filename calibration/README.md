intrinsic_calibration.py : ī�޶� intrinsic calibration �ϱ� ���� ���
 * cam_type, m, n, size, result_dir �� argument�� �Է�
 * ����(target)�� ȸ����Ű�� �̵���Ű�鼭 ������ ����(�ڵ� �Կ�)
 * results/intrinsic/<result_dir>/mtx.pkl �� calibration ���
   (camera_to_image matrix)�� �����.

make_pos_image_pairs.py : extrinsic calibration�� ���� ������ ���� ��� ���� ���
 * cam_type, data_dir �� argument�� �Է�
 * ȭ��ǥ�� W/S �� wrist�� ��ġ�� ����, Q/E/H/K/Y/I �� wrist�� rotation�� �����ϰ�,
   �����̽��ٷ� ������ ����.
 * ������ ������ ������ <data_dir>/images�� ����ǰ�, �� �������� �κ��� �ո� ��ǥ�� data_dir/poses �� �����.

extrinsic_calibration.py : ī�޶� extrinsic calibration�ϱ� ���� ���
 * cam, h, w, size, data_dir �� argument�� �Է�
 * cam�� static�� ��� : �ո� �پ��ִ� target�� static camera�� �Կ��ϴ� ��츦 ����.
                         camera_to_base matrix�� target_to_wrist matrix�� ����.

 * cam�� wrist�� ��� : �ٴڿ� ������ �ִ� target�� wrist camera�� �Կ��ϴ� ��츦 ����.
                         camera_to_wrist matrix�� target_to_base matrix�� ����.
 
 * cam�� both�� ��� : target�� wrist camera�� static camera�� ���ÿ� �Կ��ϴ� ��츦 ����.
                         static_camera_to_base matrix�� wrist_camera_to_wrist matrix�� ����.

 * intrinsic_calibration.py���� ���� ī�޶� ������, make_pos_image_pairs.py ���� ���� ������ �ֵ��� �̿���
   ������Ʈ ���� ��ǥ ��ȯ matrix�� ����
 * results/extrinsic/<cam>/mtx.pkl �� ����� �����.
   * cam�� static�� ���, camera_to_base matrix�� target_to_wrist matrix�� stack�� array�� �����.
   * cam�� wrist�� ���, camera_to_wrist matrix�� target_to_base matrix�� stack�� array�� �����.
   * cam�� both�� ���, static_camera_to_base matrix�� wrist_camera_to_wrist matrix�� stack�� array�� �����.

static_test.py : static camera�� ���� calibration�� ���� �׽�Ʈ�ϱ� ���� ���
 * h, w, size, auto�� argument�� �Է�
 * ���� ��, static camera�� target�� ���� �־�� ��(�ո� � ���� �������� �ȵ�).
 * static camera�� ���� image���� target�� ã��, ��ȯ�� ���� �� ���Ѽ� target�� base��ǥ�� ��ǥ�� ����.
 * wrist�� target�� �� ��ǥ�� �̵���Ŵ. auto == 1 �� ��� �ڵ����� �̵��ϰ�, auto == 0 �� ���
   �����̽��ٸ� ������ �̵���.
 
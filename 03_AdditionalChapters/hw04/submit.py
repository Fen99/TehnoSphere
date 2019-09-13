subm_vw = open('submission_VW(t=0.25_new).cvs', 'r')
subm_3i = open('submission_3Idiots_t(25)_l(0.000005).csv', 'r')
out = open('subm_vw(t=0.25_new)+3i_0.3(3i).csv', 'w')
out.write('Id, Click')

for line_id, lines in enumerate(zip(subm_vw, subm_3i)):
	if line_id == 0:
		continue
	else:
		line_vw, line_3i = lines
		line_vw = line_vw[:-1]
		line_3i = line_3i[:-1]
		if line_vw.split(',')[0] != line_3i.split(',')[0]:
			raise Exception('Different cases!')

		tmp = out.write(str(line_id) + "," + str(0.7 * float(line_vw.split(',')[1]) + 0.3 * float(line_3i.split(',')[1])) + '\n')
out.close()
subm_3i.close()
subm_vw.close()
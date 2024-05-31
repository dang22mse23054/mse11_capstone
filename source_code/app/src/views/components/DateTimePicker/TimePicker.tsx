import React, { FC, useEffect, Fragment } from 'react';
// import 'date-fns';
// import { vi, ja } from 'date-fns/locale';
// import DateFnsUtils from '@date-io/date-fns';
import MomentUtils from '@date-io/moment';
import { FormHelperText } from '@material-ui/core';
import MDIcon from '@mdi/react';
import { mdiAlert, mdiInformationOutline } from '@mdi/js';
import { red, grey } from '@material-ui/core/colors';
import { MuiPickersUtilsProvider, KeyboardTimePicker } from '@material-ui/pickers';
import { MaterialUiPickersDate } from '@material-ui/pickers/typings/date';
import { mdiClock } from '@mdi/js';

import moment from 'moment';
moment.locale('ja');

interface IProps {
	order: number
	hour: number
	minute: number
	variant?: 'outlined' | 'standard' | 'filled'
	size?: 'small' | 'medium'
	label?: string
	helperText?: string
	isError?: boolean
	onChange?(time: MaterialUiPickersDate): void
}

class JaMomentUtils extends MomentUtils {
	getMeridiemText(ampm) {
		return ampm === 'am' ? '午前' : '午後';
	}
}

const CustomTimePicker: FC<IProps> = (props: IProps) => {

	const hour = props.hour != null ? props.hour : 23;
	const minute = props.minute != null ? props.minute : 55;
	const [selectedTime, setSelectedTime] = React.useState<Date>(moment().set('hour', hour).set('minute', minute));
	const { size, label } = props;

	useEffect(() => {
		// update item on props changed
		const hour = props.hour != null ? props.hour : 23;
		const minute = props.minute != null ? props.minute : 55;
		setSelectedTime(moment().set('hour', hour).set('minute', minute));
	}, [props]);

	const handleChange = (time, value) => {
		setSelectedTime(time);
		if (props.onChange) {
			props.onChange(time);
		}
	};

	return (
		<Fragment>
			<MuiPickersUtilsProvider utils={JaMomentUtils} locale='ja'>

				<KeyboardTimePicker
					{...label ? { label } : {}}
					ampm={true}
					size={size || 'medium'}
					minutesStep={5}
					// autoOk={true}
					inputVariant={props.variant || 'standard'}
					value={selectedTime}
					onChange={handleChange}
					InputAdornmentProps={{ position: 'end' }}
					keyboardIcon={<MDIcon size={'18px'} path={mdiClock} />}
					KeyboardButtonProps={{
						style: { padding: 8 }
					}}
					fullWidth={true}
					okLabel="OK"
					cancelLabel="キャンセル"
				/>
				{/* 
					<TextField
						variant="outlined"
						label="Hour"
						type="time"
						defaultValue="23:00"
						InputLabelProps={{
							shrink: true,
						}}
					/> */}
			</MuiPickersUtilsProvider >
			{
				props.helperText && (
					<FormHelperText style={{ color: (props.isError ? red[500] : grey[600]), fontSize: '9pt', display: 'flex', alignItems: 'center', marginInline: 14 }}>
						<MDIcon size={'11pt'} path={props.isError ? mdiAlert : mdiInformationOutline} />&nbsp;<label>{props.helperText}</label>
					</FormHelperText>
				)
			}
		</Fragment>
	);
};

export default CustomTimePicker;
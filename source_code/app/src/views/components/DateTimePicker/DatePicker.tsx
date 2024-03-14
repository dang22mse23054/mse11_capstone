import React, { FC, Fragment, useEffect, useRef } from 'react';
import theme from 'compDir/Theme/tasktracker';

// import 'date-fns';
// import { vi, ja } from 'date-fns/locale';
// import DateFnsUtils from '@date-io/date-fns';
import moment from 'moment-timezone';
import MomentUtils from '@date-io/moment';
import { createMuiTheme, ThemeProvider } from '@material-ui/core/styles';
import { Badge, Box, FormHelperText, Tooltip } from '@material-ui/core';
import { MuiPickersUtilsProvider, KeyboardDatePicker } from '@material-ui/pickers';
import MDIcon from '@mdi/react';
import { mdiCalendar, mdiAlert, mdiInformationOutline, mdiClose } from '@mdi/js';
import CustomToolbar from './CustomToolbar';
import { green, red, grey } from '@material-ui/core/colors';
import { DateFmt } from 'constDir';
import { IHoliday } from 'interfaceDir';
import ColorButton from 'compDir/Button';
moment.locale('ja');

interface IProps {
	type?: 'date' | 'startDate' | 'endDate'
	// format yyyy-MM-dd
	size?: 'medium' | 'small'
	inputVariant: 'filled' | 'outlined' | 'standard'
	label: any
	okLabel?: string
	todayLabel?: string
	cancelLabel?: string
	helperText?: string
	isError: boolean

	value?: string
	defaultValue?: string
	clearable?: boolean
	readonly?: boolean

	holidays?: Array<IHoliday>
	disableFuture?: boolean
	disablePast?: boolean
	maxDate?: any
	minDate?: any

	onChange?(value: moment.Moment): void
	onKeyPress?(e): void
	shouldDisableDate?(date: moment.Moment): boolean
}

const custTheme = createMuiTheme({
	...theme,
	palette: {
		...theme.palette,
		secondary: {
			...theme.palette.secondary,
			main: green[600]
		}
	},
});

class JaMomentUtils extends MomentUtils {
	getMeridiemText(ampm) {
		return ampm === 'am' ? '午前' : '午後';
	}
}

const CustomDatePicker: FC<IProps> = (props: IProps) => {
	const hasMounted = useRef(false);
	const holidayBagde = (name, isToday) => (
		<Tooltip title={name || '不明'}>
			<Box fontSize={10} {...isToday && { color: 'yellow' }}>祝</Box>
		</Tooltip>
	);

	const today = moment();
	switch (props.type) {
		case 'endDate':
			today.endOf('day');
			break;

		default:
			today.startOf('day');
			break;
	}

	let date;
	if (props.value || props.defaultValue) {
		date = moment(props.value || props.defaultValue);
		if (!date) {
			date = moment();
		}
		switch (props.type) {
			case 'endDate':
				date.endOf('day');
				break;

			default:
				date.startOf('day');
				break;
		}
	} else if (props.value === null || props.defaultValue === null) {
		// if directly set value to NULL => set this NULL to value
		date = null;
	}

	// set default value
	const [selectedDate, setSelectedDate] = React.useState<moment.Moment>(date === undefined ? today : date);
	const clearable = props.clearable != null ? props.clearable : true;

	// on state changed
	useEffect(() => {
		if (!hasMounted.current) {
			// do componentDidMount logic
			hasMounted.current = true;
		} else {
			// do componentDidUpdate logic
			if (props.onChange) {
				props.onChange(selectedDate);
			}
		}
	}, [selectedDate, props.holidays]);

	const handleDateChange = (date) => {
		if (!date) {
			date = moment();
		}
		switch (props.type) {
			case 'endDate':
				date.endOf('day');
				break;

			default:
				date.startOf('day');
				break;
		}
		setSelectedDate(date);
	};

	const renderDay = (day, selectedDate, dayInCurrentMonth, dayComponent) => {
		const dayProps = dayComponent.props;
		const isHidden = dayProps.hidden;
		const isToday = today.isSame(day, 'day');
		const dayStr = day.format(DateFmt.YYYYMMDD);

		// check whether belong to holidays
		const holiday = props.holidays ? props.holidays.filter(holiday => holiday.date == dayStr)[0] : null;

		return (
			<ThemeProvider theme={custTheme}>
				{
					isHidden ? dayComponent : (
						<Badge style={{ background: 'none' }} overlap='circle'
							variant={holiday ? 'standard' : 'dot'}
							badgeContent={holiday ? holidayBagde(holiday.name, isToday) : ''}
							color={holiday ? 'error' : isToday ? 'secondary' : 'default'} >{dayComponent}</Badge>
					)
				}
			</ThemeProvider>
		);
	};

	return (
		<Fragment>
			<MuiPickersUtilsProvider utils={JaMomentUtils} locale={'ja'}>
				<KeyboardDatePicker disableToolbar={false} allowKeyboardControl
					autoOk={false}
					variant='dialog'
					fullWidth={true}
					size={props.size || 'medium'}
					inputVariant={props.inputVariant || 'standard'}

					// https://github.com/mui/material-ui-pickers/issues/1156#issuecomment-669893593
					InputProps={{
						startAdornment: clearable && selectedDate && props.disabled != true && (
							<ColorButton onClick={() => setSelectedDate(null)} btnType='icon' size='small' style={{ order: 1 }}>
								<MDIcon color={grey[600]} size={'20px'} path={mdiClose} />
							</ColorButton>
						),
						readOnly: props.readonly
					}}
					InputAdornmentProps={{
						position: 'end',
						style: { order: 2, marginLeft: 0 }
					}}

					ToolbarComponent={(props) => <CustomToolbar {...props} />}
					// margin="normal"
					renderDay={renderDay}
					onChange={handleDateChange}
					views={['year', 'month', 'date']}
					keyboardIcon={<MDIcon size={'18px'} path={mdiCalendar} />}
					KeyboardButtonProps={{
						style: { padding: 8 }
					}}
					maskChar='_'
					format="YYYY/MM/DD"
					invalidDateMessage='無効な日付です'
					maxDateMessage={`Max date is ${props.maxDate}`}
					showTodayButton={true}
					todayLabel="今日"
					okLabel="OK"
					cancelLabel="キャンセル"
					label={props.label}
					value={props.value || selectedDate}

					{...props.onKeyPress && { onKeyPress: props.onKeyPress }}
					{...props.disabled != null && { disabled: props.disabled }}
					{...props.disableFuture && { disableFuture: props.disableFuture }}
					{...props.disablePast && { disablePast: props.disablePast }}
					{...props.maxDate && { maxDate: props.maxDate }}
					{...props.minDate && { minDate: props.minDate }}
					{...props.shouldDisableDate && { shouldDisableDate: props.shouldDisableDate }}
				/>
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


export default CustomDatePicker;
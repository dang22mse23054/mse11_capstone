import React, { Fragment, Component } from 'react';
import { ButtonGroup, FormControl, FormLabel, Grid, MenuItem, Select, Switch } from '@material-ui/core';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import { blue, grey, red } from '@material-ui/core/colors';
import ColorButton from 'compDir/Button';
import { IFrequency as WeekDateValue } from 'interfaceDir';
import {
	TypeNames, Types, WeekLabels, WeekOfMonth,
	DayOfWeek, DayOfWeekLabels, DayOfWeekNames
} from 'constDir';

export interface IState extends WeekDateValue {
}

interface IProps {
	data?: WeekDateValue
	onChange?(state: IState): void
}

const useStyles = (theme: Theme) => createStyles({
	circleBtn: {
		width: 36,
		margin: '0 2px',
		textAlign: 'center',
		'& button': {
			// border-radius: 50%;
			// border: 1px solid red;
			// min-width: inherit;
			// line-height: initial;
			// padding: 5 10
			borderRadius: '50%',
			// border: `solid ${grey[400]} 1px`,
			minWidth: 'inherit',
			lineHeight: 'initial',
			padding: '5px 10px',
			width: 36,
			height: 36,
		}
	},
	week: {
		minWidth: 60,
	},
	placeholder: {
		color: grey[500]
	}
});

class WeekDatePicker extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
		data: {
			type: TypeNames.MONTHLY_BY_WEEK,
			weekOfMonth: '',
			dayOfWeek: '',
			dates: '',
			weekOfMonthArray: [],
			dayOfWeekArray: [],
			datesArray: [],
			// fixedDate: this.props.data?.fixedDate,
			isOnlyBusinessDays: false,
			isSkipHoliday: false,
			autoScheduling: 0
		}
	}
	// Set default state
	// public state: IState = {
	// 	type: this.props.data?.type || TypeNames.MONTHLY_BY_WEEK,
	// 	weekOfMonth: this.props.data?.weekOfMonth || '',
	// 	dayOfWeek: this.props.data?.dayOfWeek || '',
	// 	dates: this.props.data?.dates || '',
	// weekOfMonthArray: this.props.data?.weekOfMonth?.split(',') || [],
	// dayOfWeekArray: this.props.data?.dayOfWeek?.split(',') || [],
	// datesArray: this.props.data?.dates?.split(',') || [],
	// 	fixedDate: this.props.data?.fixedDate,
	// 	isOnlyBusinessDays: this.props.data?.isOnlyBusinessDays == null ? false : this.props.data?.isOnlyBusinessDays,
	// 	isSkipHoliday: this.props.data?.isSkipHoliday == null ? false : this.props.data?.isSkipHoliday,
	// 	autoScheduling: this.props.data?.autoScheduling || 0
	// }

	componentDidUpdate = () => {
		// if (this.props.onChange) {
		// 	this.props.onChange(this.state);
		// }
	}

	onChangeType = (e) => {
		const type = Number(e.target.value);
		const newState = {
			...this.props.data,
			type,
			// weekOfMonth: '',
			// dayOfWeek: '',
			// dates: '',
			// weekOfMonthArray: [],
			// dayOfWeekArray: [],
			// datesArray: [],
			// isOnlyBusinessDays: false,
		};
		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}

	onToggleBusinessDay = (e) => {
		const newState = {
			...this.props.data,
			dates: '',
			datesArray: [],
			isOnlyBusinessDays: !this.props.data?.isOnlyBusinessDays
		};

		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}

	onToggleSkipHoliday = (e) => {
		const newState = {
			...this.props.data,
			isSkipHoliday: !this.props.data.isSkipHoliday
		};
		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}

	onClickDate = (value) => {
		let datesArray = [...this.props.data.datesArray];

		if (datesArray.includes(value)) {
			datesArray = datesArray.filter(date => value != date);
		} else {
			datesArray.push(value);
			datesArray = datesArray.sort((a, b) => a - b);
		}

		// sort list again
		datesArray.sort((a, b) => a - b);

		const newState = {
			...this.props.data,
			datesArray,
			dates: datesArray.toString()
		};

		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}

	onClickWeek = (value) => {
		let weekOfMonthArray = [...this.props.data.weekOfMonthArray];

		if (weekOfMonthArray.includes(value)) {
			weekOfMonthArray = weekOfMonthArray.filter(week => value != week);
		} else {
			weekOfMonthArray.push(value);
			weekOfMonthArray = weekOfMonthArray.sort((a, b) => a - b);
		}

		const newState = {
			...this.props.data,
			weekOfMonthArray,
			weekOfMonth: weekOfMonthArray.toString()
		};

		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}

	onClickDayOfWeek = (value) => {
		let dayOfWeekArray = [...this.props.data.dayOfWeekArray];

		if (dayOfWeekArray.includes(value)) {
			dayOfWeekArray = dayOfWeekArray.filter(day => value != day);
		} else {
			dayOfWeekArray.push(value);
			dayOfWeekArray = dayOfWeekArray.sort((a, b) => a - b);
		}

		const newState = {
			...this.props.data,
			dayOfWeekArray,
			dayOfWeek: dayOfWeekArray.toString()
		};

		if (this.props.onChange) {
			this.props.onChange(newState);
		}
	}



	renderDate = (value: WeekDateValue) => {
		const { classes } = this.props;
		// 1~31 is date of month,
		// 1~23 is business day
		const maxItems = value.isOnlyBusinessDays ? 23 : 31;
		const days = [...Array(maxItems).keys()].map((i, idx) => i + 1);
		// but 32 is 最終日 (the last day of Month/Business_day)
		days.push(32);

		return (
			<Grid item>
				<FormControl style={{ width: 155, marginInline: 20 }} >
					<Select multiple value={this.props.data?.datesArray?.length > 0 ? this.props.data?.datesArray : ['unset']}
						classes={this.props.data?.datesArray?.length > 0 ? {} : { select: classes.placeholder }}
					// onChange={handleChange}
					>
						{
							days.map((value, id) =>  {
								// const label = `${value}${value == 32 ? '(最終日)' : ''}`;
								const label = value == 32 ? '最終日' : value;
								return (
									<MenuItem key={id} value={value} onClick={() => this.onClickDate(value)}>
										{label}
									</MenuItem>
								);
							})
						}
						{
							this.props.data?.datesArray?.length == 0 && (
								<option value="unset" disabled hidden>日</option>
							)
						}
					</Select>
				</FormControl>
			</Grid >

		);

		// const rows = [];
		// while (days.length > 0) {
		// 	rows.push(days.splice(0, 7));
		// }
		// return rows.map((row, index) => {
		// 	const emptyItems = [...Array(7 - row.length).keys()];
		// 	return (
		// 		<Grid key={index} item container style={{ flexBasis: '100%' }} justify='center'>
		// 			{
		// 				row.map((value, id) => {
		// 					return (
		// 						<div key={id} className={classes.circleBtn} >
		// 							<ColorButton onClick={() => this.onClickDate(value)}
		// 								variant={this.props.data?.datesArray.includes(value) ? 'contained' : 'outlined'}
		// 								btnColor={this.props.data?.datesArray.includes(value) ? 'secondary' : 'grey'}
		// 								btnContrast={this.props.data?.datesArray.includes(value) ? 600 : 500}
		// 							>{value}</ColorButton>
		// 						</div>
		// 					);
		// 				})
		// 			}
		// 			{
		// 				emptyItems.map((item, id) => {
		// 					return (
		// 						<div key={id} className={classes.circleBtn}></div>
		// 					);
		// 				})
		// 			}
		// 		</Grid>
		// 	);
		// });
	}

	renderWeek = (type: typeof Types[number], value: WeekDateValue) => {
		const { classes } = this.props;

		return (
			<Fragment>
				{
					type != TypeNames.WEEKLY && (
						<Grid item container style={{ flexBasis: '100%' }} justify='center'>
							<ButtonGroup size="small">
								{
									WeekOfMonth.map((key, index) => (
										<ColorButton className={classes.week} key={index}
											onClick={() => this.onClickWeek(key)}
											variant={this.props.data?.weekOfMonthArray?.includes(key) ? 'contained' : 'outlined'}
											btnColor={this.props.data?.weekOfMonthArray?.includes(key) ? 'secondary' : 'grey'}
											btnContrast={this.props.data?.weekOfMonthArray?.includes(key) ? 600 : 500}
										>{WeekLabels[key]}</ColorButton>
									))
								}
							</ButtonGroup>
						</Grid>
					)
				}
				<Grid item container style={{ flexBasis: '100%' }} justify='center'>
					{
						DayOfWeek.map((key, index) => {
							const isSelected = this.props.data?.dayOfWeekArray.includes(key);
							const style = {};

							if (!isSelected) {
								switch (key) {
									case DayOfWeekNames.SAT:
										style.color = blue[600];
										break;
									case DayOfWeekNames.SUN:
										style.color = red[600];
										break;
								}
							}

							return (
								<div className={classes.circleBtn} key={index}>
									<ColorButton onClick={() => this.onClickDayOfWeek(key)}
										style={style}
										variant={isSelected ? 'contained' : 'outlined'}
										btnColor={isSelected ? 'secondary' : 'grey'}
										btnContrast={isSelected ? 600 : 500}
									>{DayOfWeekLabels[key]}</ColorButton>
								</div>
							);
						})
					}
				</Grid>
			</Fragment>
		);
	}

	public render = (): React.ReactNode => {
		const type = this.props.data?.type || TypeNames.MONTHLY_BY_WEEK;
		const {
			weekOfMonthArray, dayOfWeekArray, datesArray, fixedDate,
			isOnlyBusinessDays, isSkipHoliday, autoScheduling
		} = this.props.data;

		return (
			<Grid container spacing={2} wrap='nowrap' justify='flex-end'>
				{/* <Grid item style={{ flex: 1 }}>
							<FormControl variant="outlined" fullWidth={true} size={'small'} >
								<InputLabel id="week-date-type">タイプ</InputLabel>
								<Select onChange={onChangeType} labelId="week-date-type" label="タイプ" value={type}>
									{
										Types.map((key, index) => (
											<MenuItem key={index} value={key}>{TypeLabels[key]}</MenuItem>
										))
									}
								</Select>
							</FormControl>
	
						</Grid> */}
				<Grid item style={{ flex: 1 }}>
					<Grid container spacing={1} style={{ borderColor: grey[400], }} justify='flex-end'>
						{
							type == TypeNames.MONTHLY_BY_DATE ? this.renderDate({
								type, weekOfMonthArray, dayOfWeekArray, datesArray, fixedDate,
								isOnlyBusinessDays, isSkipHoliday, autoScheduling,
							}) : this.renderWeek(type, {
								type, weekOfMonthArray, dayOfWeekArray, datesArray, fixedDate,
								isOnlyBusinessDays, isSkipHoliday, autoScheduling,
							})
						}
					</Grid>
				</Grid>
				<Grid item>
					<Grid container spacing={2}>
						{
							type == TypeNames.MONTHLY_BY_DATE && (
								<Grid item>
									<FormControl>
										<FormLabel style={{ fontSize: '0.75em' }}>営業日のみ</FormLabel>
										<Switch size={'small'} color="primary" disabled={type != TypeNames.MONTHLY_BY_DATE}
											checked={isOnlyBusinessDays} onChange={this.onToggleBusinessDay} />
									</FormControl>
								</Grid>
							)
						}
						<Grid item>
							<FormControl>
								<FormLabel style={{ fontSize: '0.75em' }}>祝日を除く</FormLabel>
								<Switch size={'small'} color="primary"
									checked={isSkipHoliday} onChange={this.onToggleSkipHoliday} />
							</FormControl>
						</Grid>
					</Grid>
				</Grid>

			</Grid>
		);
	}
}

export default withStyles(useStyles)(WeekDatePicker);
import React, { memo, Component, RefObject } from 'react';
import { theme, extColors, basic } from 'stylesDir/colors';
import border from 'stylesDir/border.module.css';
import { customStaticRanges } from './static-range';
import 'react-date-range/dist/styles.css'; // main css file
import 'react-date-range/dist/theme/default.css'; // theme css file
import { DateRangePicker } from 'react-date-range';
import * as locales from 'react-date-range/dist/locale';
import { FormControl, Popover, TextField } from '@material-ui/core';
import MDIcon from '@mdi/react';
import { mdiMenuDown, mdiMenuUp } from '@mdi/js';
import moment from 'moment';

export interface DropDownOption {
	isDefault?: boolean
	label: string
	value: string | number
	to?: string
}

export interface IState {
	range?: any
	searchDateRange?: string
	datePickerOn: boolean
	isEnabled?: boolean
	btnRef?: RefObject<any>
	inputRef?: RefObject<any>
	// store the from date before saving
	constFromDate: Date
	minDate: Date
}

interface IProps {
	label?: string;
	value?: string;
	startDate?: Date;
	fixedStartDate?: boolean;
	endDate?: Date;
	fixedEndDate?: boolean;
	btnColor?: string
	fontSize?: number
	fontColor?: string
	borderRadius?: number
	isCloseAfterClick?: boolean
	width: number | string
	btnWidth?: number | string
	isLeft: boolean
	isEnabled: boolean
	options: Array<DropDownOption>
	onChange(item): any
	onClose?(item): any
	onSwitchEnable?(isEnabled, item): any
}

class CustomDateRangePicker extends Component<IProps, IState> {
	private timer = -1;

	constructor(props: IProps) {
		super(props);

		const beginToday = moment().startOf('day').milliseconds(0).toDate();
		const endToday = moment().endOf('day').milliseconds(0).toDate();
		const startDate = moment(this.props.startDate).startOf('day').milliseconds(0).toDate();
		const endDate = moment(this.props.endDate).endOf('day').milliseconds(0).toDate();

		const constFromDate = (startDate < beginToday ? startDate : beginToday);

		let minDate = constFromDate;
		// Case: future form -> closed -> reopen -> fix minDate to be to future startDate
		if (this.props.fixedStartDate && startDate > beginToday) {
			minDate = startDate;
		}

		const state: IState = {
			range: {
				startDate: this.props.startDate,
				endDate: endDate < endToday ? endToday : endDate,
				key: 'selection',
			},
			searchDateRange: '',
			datePickerOn: false,
			isEnabled: this.props.isEnabled == null ? true : this.props.isEnabled,
			inputRef: React.createRef(),
			constFromDate,
			minData: minDate
		};

		// Set default state
		this.state = state;

		this.props.onChange(this.state.range);
	}

	// Set default properties's values
	public static defaultProps: Partial<IProps> = {
		btnColor: extColors.blueDark,
		fontSize: 12,
		fontColor: basic.darkGray1,
		borderRadius: 2,
		isCloseAfterClick: true,
		width: 'auto',
		btnWidth: 'auto',
		isLeft: true,
		isEnabled: true,
		options: [],
		startDate: moment().startOf('day').toDate(),
		endDate: moment().endOf('day').toDate(),
		onChange: (range) => false,
		onClose: (range) => false,
		onSwitchEnable: (isEnabled, range) => false,
	}


	componentDidMount = () => {
		this.setDateLabel();
	}

	handleSelect = async (range) => {
		range = range[this.state.range.key];
		const beginToday = moment().startOf('day').milliseconds(0).toDate();
		const endToday = moment().endOf('day').milliseconds(0).toDate();
		const startDate = moment(range.startDate).startOf('day').milliseconds(0).toDate();
		const endDate = moment(range.endDate).endOf('day').milliseconds(0).toDate();
		range = {
			...range,
			startDate: this.props.fixedStartDate ? this.props.startDate : (startDate < beginToday ? this.state.constFromDate : startDate),
			endDate: this.props.fixedEndDate ? this.props.endDate : (endDate < endToday ? endToday : endDate)
		};
		await this.setState({ range });
		this.setDateLabel();
		this.props.onChange(this.state.range);
	}

	setDateLabel = () => {
		const startDate = this.state.range.startDate;
		const endDate = this.state.range.endDate;

		const now = moment().format('YYYY/MM/DD');
		const from = moment(startDate).format('YYYY/MM/DD');
		let to = moment(endDate).format('YYYY/MM/DD');
		to = to > now ? to : now;
		this.setState({ searchDateRange: from == to ? from : `${from} ~ ${to}` });
	}

	// public keyPress = async ({ nativeEvent }) => {
	// 	if (nativeEvent.key === 'Backspace') {
	// 		await this.setState({
	// 			range: {
	// 				startDate: moment().startOf('month').toDate(),
	// 				endDate: moment().endOf('day').toDate(),
	// 				key: 'selection',
	// 			}
	// 		});
	// 		this.setDateLabel();
	// 	}
	// }

	public openPicker = (event) => {
		if (!this.state.datePickerOn) {
			if (!this.state.isEnabled) {
				this.onSwitchEnable();
			}
			this.setState({
				datePickerOn: !this.state.datePickerOn,
				btnRef: event.currentTarget
			});
		}
	}

	public closePicker = () => {
		if (this.state.datePickerOn) {
			Promise.resolve()
				.then(res => this.setState({
					datePickerOn: false,
					btnRef: undefined
				}))
				.then(res => this.props.onClose!(res))
				.catch((e) => console.error(e));
		}
	}

	public onSwitchEnable = () => {
		this.timer = window.setTimeout(() => {
			Promise.resolve()
				.then(res => this.setState({ isEnabled: !this.state.isEnabled }))
				.then(res => this.props.onSwitchEnable!(isEnabled, this.state.range))
				.catch((e) => console.error(e));
		}, 10);
	}

	public disablePicker = (e) => {
		e.preventDefault();
		e.stopPropagation();
		this.setState({ datePickerOn: false });
		this.onSwitchEnable();
	}

	public render(): React.ReactNode {

		const style: any = {
			position: 'absolute',
			top: '100%',
			backgroundColor: '#ffffff',
		};
		style[this.props.isLeft ? 'left' : 'right'] = 0;

		return (
			<FormControl ref={this.state.inputRef} fullWidth={true}>
				<TextField style={{ cursor: 'pointer' }} onClick={this.openPicker}
					variant="outlined"
					label={this.props.label}
					InputLabelProps={{ shrink: true }}
					InputProps={{
						inputComponent: ({ className }) => (
							<div style={{ padding: 8, paddingLeft: 14, paddingRight: 7, width: '100%', cursor: 'pointer' }}>
								<div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
									<div style={{ /* textAlign: 'center',  */width: '100%' }}>
										{this.state.isEnabled ? this.state.searchDateRange : this.props.label}
									</div>
									<span style={{ display: 'flex' }}>
										{/* {
											!this.state.datePickerOn && this.state.isEnabled &&
											<span style={{ display: 'flex', alignItems: 'center' }}
												onClick={this.disablePicker}>
												<MDIcon color={colors.red} size={'14px'} path={mdiClose} />
											</span>
										} */}
										<MDIcon color={this.props.fontColor} size={1} path={this.state.datePickerOn ? mdiMenuUp : mdiMenuDown} />
									</span>
								</div>
							</div>
						)
					}}
				/>

				<Popover
					open={this.state.datePickerOn}
					anchorEl={this.state.btnRef as any}
					anchorOrigin={{
						vertical: 'bottom',
						horizontal: 'left',
					}}
					transformOrigin={{
						vertical: 'top',
						horizontal: 'left',
					}}
					onClose={(event, reason) => {
						if (reason == 'backdropClick') {
							this.closePicker();
						}
					}}>

					<DateRangePicker className={border.style}
						showSelectionPreview={true}
						editableDateInputs={true}
						// moveRangeOnFirstSelection={false}
						// retainEndDateOnFirstSelection={false}
						ranges={[this.state.range]}
						onChange={this.handleSelect}
						locale={locales['ja']}
						dateDisplayFormat='yyyy年MM月dd日'
						inputRanges={[]}
						staticRanges={customStaticRanges}
						rangeColors={[theme.vermillion]}
						minDate={this.state.minData}
						// maxDate={this.props.fixedEndDate ? this.props.endDate : null}
						showDateDisplay={true}
						scroll={{ enabled: true }}
					/>
				</Popover>
			</FormControl>
		);
	}
}

export default memo(CustomDateRangePicker);

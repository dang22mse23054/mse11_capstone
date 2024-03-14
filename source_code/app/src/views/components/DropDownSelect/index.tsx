import React, { Component } from 'react';
import { grey, red } from '@material-ui/core/colors';
import MDIcon from '@mdi/react';
import ColorButton from 'compDir/Button';
import { FormControl, MenuItem, TextField, Grid, InputAdornment, FormHelperText } from '@material-ui/core';
import { mdiMenuUp, mdiMenuDown, mdiClose } from '@mdi/js';

export interface DropDownOption {
	isDefault?: boolean
	label: string
	value: any
	iconName?: string
	iconColor?: string
	iconActiveColor?: string
	iconFixedColor?: string
}

export interface IState {
	selectedItem?: DropDownOption
	activeIndex?: number
	defaultValue?: number
}

interface IProps {
	label?: string
	disabled?: boolean
	isError?: boolean
	isShrink?: boolean
	autoSetDefautlValue?: boolean
	mustSelect?: boolean
	placeholder?: string;
	value?: any
	variant?: 'filled' | 'outlined' | 'standard'
	size?: 'medium' | 'small'
	options: Array<DropDownOption>
	helperText?: string
	onChange(item): any
}

class DropDownSelect extends Component<IProps, IState> {

	constructor(props) {
		super(props);
	}

	// Set default properties's values
	public static defaultProps: Partial<IProps> = {
		isError: false,
		autoSetDefautlValue: false,
		variant: 'standard',
		size: 'medium',
		options: [],
		mustSelect: true,
		disabled: false,
	}

	public state: IState = {
		selectedItem: null,
		activeIndex: ''
	}

	componentDidMount() {
		this.initValue();
	}

	async componentDidUpdate(prevProps: IProps, prevState: IState, snapshot) {
		//if (prevProps.value != this.props.value) {
		//	this.initValue();
		//}
		// NaN is not equal itself. Comparing value by equal operator then NaN value will cause infinite render 
		// => Use Object.is instead for safe.
		if (!Object.is(prevProps.value, this.props.value)) {
			this.initValue();
		}
	}

	initValue = () => {
		const props = this.props;
		const state: any = {
			selectedItem: null,
			activeIndex: ''
		};

		for (let index = 0; index < props.options.length; index++) {
			const item = props.options[index];

			if (item.value == this.props.value) {
				state.selectedItem = item;
				state.activeIndex = index;
				break;

			} else if (this.props.autoSetDefautlValue && item.isDefault) {
				state.selectedItem = item;
				state.activeIndex = index;
			}

		}

		// Set default state
		this.setState(state);
	}


	public onSelect = async (item, index) => {
		this.setState({
			selectedItem: item,
			activeIndex: index
		});
		if (this.props.onChange) {
			return await Promise.resolve()
				.then(res => this.props.onChange(item))
				.catch((e) => console.error(e));
		}
		// return false;
	}


	public clearSelectedValue = async () => {
		this.setState({
			selectedItem: null,
			activeIndex: ''
		});

		if (this.props.onChange) {
			return await Promise.resolve()
				.then(res => this.props.onChange(null))
				.catch((e) => console.error(e));
		}
	}

	public render(): React.ReactNode {
		return (
			<FormControl fullWidth={true}>
				<TextField select disabled={this.props.disabled}
					InputLabelProps={{ shrink: this.props.isShrink }}
					SelectProps={{
						IconComponent: ({ className }) => {
							const isOpen = className.includes('MuiSelect-iconOpen');
							const hasValue = this.state.activeIndex?.toString() != '';
							return (
								<div style={{ position: 'absolute', right: 9 }}>
									<InputAdornment position="end">
										{
											!this.props.mustSelect && hasValue && (
												<ColorButton size='small' btnType="icon" btnColor='grey' btnContrast={600}
													onClick={this.clearSelectedValue}>
													<MDIcon size={'20px'} path={mdiClose} />
												</ColorButton>
											)
										}
										<MDIcon color={grey[600]} size={1} path={isOpen ? mdiMenuUp : mdiMenuDown} />
									</InputAdornment>
								</div>
							);
						}
					}}
					onChange={(event, child) => {
						const index = event.target.value;
						this.onSelect(this.props.options[index as any], index);
					}}

					error={this.props.isError}
					size={this.props.size}
					variant={this.props.variant}
					label={this.props.label}
					value={this.state.activeIndex} >

					{
						this.props.options.map((item, index) => {
							const isActivating = this.state.activeIndex == index;
							return (
								<MenuItem value={index} key={`item_${index}`}>
									<Grid container spacing={1} alignItems='center'>
										{
											item.iconName && (
												<Grid item style={{ display: 'flex', marginRight: 8 }}>
													<MDIcon size={'13pt'} path={item.iconName}
														color={item.iconFixedColor || (isActivating ? (this.state.iconActiveColor || grey[800]) : item.iconColor)} />
												</Grid>
											)
										}
										<Grid item>{item.label}</Grid>
									</Grid>
								</MenuItem>
							);
						})
					}
				</TextField>

				{
					this.props.helperText && (
						<FormHelperText style={{ color: (this.props.isError ? red[500] : grey[600]), fontSize: '10pt', display: 'flex', alignItems: 'center', marginInline: 0 }}>
							<label>{this.props.helperText}</label>
						</FormHelperText>
					)
				}
			</FormControl>
		);
	}
}

export default DropDownSelect;
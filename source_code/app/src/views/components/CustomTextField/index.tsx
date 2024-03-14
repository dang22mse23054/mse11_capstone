
import React, { Component } from 'react';
import MDIcon from '@mdi/react';
import { red, grey } from '@material-ui/core/colors';
import { mdiAlert, mdiInformationOutline } from '@mdi/js';

import { createStyles, withStyles, Theme } from '@material-ui/core/styles';
import {
	TextField, FormControl, FormHelperText
} from '@material-ui/core';
import { IAnswerProps, IAnswerStates } from 'compDir/Answer/interface';


interface IState extends IAnswerStates {
	answerValue: string,
	isRequired: boolean
}

interface IProps extends IAnswerProps {
	onChange?(answerValue: string): any
	isMultiple?: boolean
	isRequired?: boolean
	isShrink?: boolean
	prefix: string
	variant?: 'filled' | 'outlined' | 'standard'
	size?: 'medium' | 'small'
	defaultValue?: string
	placeholder?: string
	helperText?: string
	isError?: boolean
	classes?: any
}


const useStyles = (theme: Theme) =>
	createStyles({
		formControl: {
			// margin: theme.spacing(2),
		},
	});

class TextAnswer extends Component<IProps, IState> {

	constructor(props) {
		super(props);
	}

	// Set default state
	public state: IState = {
		answerValue: this.props.defaultValue ? this.props.defaultValue : '',
		isRequired: this.props.isRequired ? this.props.isRequired : false,
	}

	// public static defaultProps: Partial<IProps> = {}
	public static defaultProps: IProps = {
		isMultiple: false,
		isError: false,
		isShrink: true,
		variant: 'outlined',
		size: 'medium',
		placeholder: '',
		label: '回答',
		prefix: 'text'
	}

	handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const answerValue = (event.target as HTMLInputElement).value;
		this.setState({ answerValue });
		if (this.props.onChange) {
			this.props.onChange(answerValue);
		}
	}

	public render = (): React.ReactNode => {
		const { classes } = this.props;

		return (
			<FormControl variant="outlined" fullWidth={true} size={'small'} className={classes.formControl}>
				<TextField InputLabelProps={this.props.isShrink == null ? {} : { shrink: this.props.isShrink }}
					autoComplete='off'
					id={`${this.props.prefix}_answer`}
					name={`${this.props.prefix}_answer`}
					variant="outlined"
					label={this.props.label}
					placeholder={this.props.placeholder}
					multiline={this.props.isMultiple}
					rows={2}
					rowsMax={12}
					onChange={this.handleChange}
					value={this.state.answerValue}
					error={this.props.isError}
				/>
				{
					this.props.helperText && (
						<FormHelperText style={{ color: (this.props.isError ? red[500] : grey[600]), fontSize: '9pt', display: 'flex', alignItems: 'center' }}>
							<MDIcon size={'11pt'} path={this.props.isError ? mdiAlert : mdiInformationOutline} />&nbsp;<label>{this.props.helperText}</label>
						</FormHelperText>
					)
				}
			</FormControl>
		);
	}
}

export default withStyles(useStyles)(TextAnswer);

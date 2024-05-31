import React, { Component, CSSProperties, Fragment, RefObject } from 'react';
import ColorButton from 'compDir/Button';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import Dialog, { DialogProps } from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import { CircularProgress, Divider, Typography } from '@material-ui/core';

import Slide from '@material-ui/core/Slide';
import Grid from '@material-ui/core/Grid';
import MDIcon from '@mdi/react';
import { mdiWindowClose } from '@mdi/js';

export interface IState {
	open: boolean
	scroll: DialogProps['scroll']
	descriptionElementRef?: RefObject<any>
}

interface IProps {
	title?: string
	maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | false
	fullWidth?: boolean
	style?: CSSProperties
	content?: string
	submitLabel?: string
	submitBtnColor?: string
	isSubmitting?: boolean
	closeBtn?: boolean
	closeLabel?: string
	closeBtnColor?: string
	extBtn?: boolean
	extBtnLabel?: string
	extBtnColor?: string
	extBtnVariant?: 'contained' | 'outlined' | 'text'
	divider?: boolean | {
		top: boolean
		bottom: boolean
	}
	justifyActionBtn?: string
	slide?: string
	custClasses?: any
	handleClose?(): any
	handleSubmit?(event?: React.ChangeEvent<HTMLInputElement>): any
	handleExtBtn?(): any
}

const useStyles = (theme: Theme) => createStyles({
	root: {
		margin: 0,
		padding: theme.spacing(2),
	},
	closeButton: {
		position: 'absolute',
		right: theme.spacing(1),
		top: theme.spacing(1),
		color: theme.palette.grey[500],
	},
});


class BaseModal extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
		// submitLabel: 'Apply',
		maxWidth: 'sm',
		fullWidth: false,
		closeBtn: true,
		closeLabel: 'Close',
		submitBtnColor: 'primary',
		closeBtnColor: 'secondary',
		isSubmitting: false,
		extBtn: false,
		extBtnColor: 'secondary',
		extBtnVariant: 'text',
		divider: true,
		justifyActionBtn: 'flex-end',
		handleClose: () => false,
		handleSubmit: () => false,
		handleExtBtn: () => false
	}

	// Set default state
	public state: IState = {
		scroll: 'paper',
		descriptionElementRef: React.createRef()
	}

	componentDidMount() {
		if (this.state.open) {
			const { current: descriptionElement } = this.state.descriptionElementRef;
			if (descriptionElement !== null) {
				descriptionElement.focus();
			}
		}
	}

	transition = this.props.slide ? React.forwardRef((props, ref) => <Slide direction={this.props.slide} in ref={ref} {...props} />) : null;

	public render(): React.ReactNode {
		const content = this.props.children || this.props.content;
		const { style, classes, custClasses } = this.props;
		const divider = this.props.divider;

		return (
			<Dialog
				classes={custClasses}
				open={true}
				{...(this.transition ? { TransitionComponent: this.transition } : {})}
				maxWidth={this.props.maxWidth}
				fullWidth={this.props.fullWidth}
				onClose={this.handleClose}
				scroll={this.state.scroll}>
				{this.props.title != null && (
					<Fragment>
						<DialogTitle disableTypography className={classes.root}>
							<Typography component='div' variant="h6">{this.props.title}</Typography>
							{
								this.props.closeBtn ? (
									<ColorButton disabled={this.props.isSubmitting} btnType='icon' className={classes.closeButton} onClick={this.props.handleClose}>
										<MDIcon size={'24px'} path={mdiWindowClose} />
									</ColorButton>
								) : null
							}
						</DialogTitle>
						{divider && (divider == true || divider.top == true) && <Divider />}
					</Fragment>
				)}
				{
					content != null && (
						<Fragment>
							<DialogContent dividers={scroll === 'paper'} {...{ style }}>
								{
									this.props.content ? (
										<DialogContentText style={{ whiteSpace: 'pre-wrap' }} id="scroll-dialog-description" ref={this.state.descriptionElementRef} tabIndex={-1}>
											{content}
										</DialogContentText>
									) : (
										<Fragment>{content}</Fragment>
									)
								}

							</DialogContent>
							{divider && (divider == true || divider.bottom == true) && <Divider />}
						</Fragment>
					)
				}
				<DialogActions>
					<Grid container spacing={1} alignItems='center' justify={this.props.justifyActionBtn}>
						<Grid item>
							{this.props.extBtn && (
								<ColorButton disabled={this.props.isSubmitting} btnColor={this.props.extBtnColor} variant={this.props.extBtnVariant}
									onClick={this.props.handleExtBtn}>{this.props.extBtnLabel}</ColorButton>
							)}
						</Grid>
						<Grid item>
							<Grid container spacing={1}>
								<Grid item>
									<ColorButton disabled={this.props.isSubmitting} btnColor={this.props.closeBtnColor} onClick={this.props.handleClose} >{this.props.closeLabel}</ColorButton>
								</Grid>
								<Grid item>
									{this.props.submitLabel && (<ColorButton disabled={this.props.isSubmitting} btnColor={this.props.submitBtnColor} onClick={this.props.handleSubmit} variant="contained">
										{this.props.submitLabel}</ColorButton>)}
								</Grid>
							</Grid>
						</Grid>
					</Grid>

				</DialogActions>
			</Dialog>
		);
	}
}

export default withStyles(useStyles)(BaseModal);

